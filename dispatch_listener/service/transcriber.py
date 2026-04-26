"""Whisper transcription — remote (HTTP server) primary, local (whisper.cpp) fallback.

Two modes (auto-selected based on whisper_server_url):

- REMOTE (preferred): POST audio to a faster-whisper HTTP server (typically
  whisper-large-v3 on a GPU box reachable via Tailscale). Best accuracy,
  bounded latency. Falls back to LOCAL on any HTTP/connection error.

- LOCAL: pywhispercpp (whisper.cpp under the hood) running in-addon on the
  HA Yellow CPU. Lightweight model only (tiny.en / base.en). Fallback for
  when the remote server is unreachable.

Optional acoustic preprocessing (ffmpeg filter chain) can run before either
path. ONLY applies to Whisper-bound audio — the DTMF decoder always reads
the raw stream so tone detection is unaffected by these settings.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from pathlib import Path

import httpx
import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("/data/models")

# ffmpeg filter chain for radio voice — see #3 in design notes
PREPROCESS_FILTER_CHAIN = (
    "highpass=f=300,"
    "lowpass=f=3400,"
    "afftdn=nf=-25,"
    "compand=attacks=0.05:decays=0.4:"
    "points=-90/-90|-30/-15|-10/-7|0/-3:soft-knee=6:gain=0,"
    "loudnorm=I=-16:TP=-1.5:LRA=7"
)

# Whisper boilerplate hallucinations on near-silent / pure-noise audio.
# Source: empirical — these all show up in our actual server logs on bursts
# that contain just static or very short noise. Reject transcripts whose
# normalized form matches any of these so the prealert matcher doesn't see
# garbage. Add new strings here as they're observed in the wild.
WHISPER_HALLUCINATIONS = {
    "subtitles by the amara.org community",
    "subtitles by the amara org community",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thank you.",
    "thank you",
    "you",
    ".",
    "okay.",
    "okay",
    "bye.",
    "bye",
    "[music]",
    "[applause]",
    "subtitles",
}


def _is_hallucination(text: str) -> bool:
    if not text:
        return False
    norm = text.strip().lower().rstrip(".!? ")
    if norm in WHISPER_HALLUCINATIONS:
        return True
    # Also catch "subtitles by the amara.org community" with prefix/suffix noise
    if "amara.org" in norm or "amara org" in norm:
        return True
    return False


class Transcriber:
    def __init__(
        self,
        model_name: str = "base.en",
        preprocess: bool = True,
        server_url: str = "",
        server_timeout_sec: float = 30.0,
        initial_prompt_provider=None,
    ) -> None:
        self.model_name = model_name
        self.preprocess = preprocess
        self.server_url = server_url.rstrip("/") if server_url else ""
        self.server_timeout_sec = server_timeout_sec
        # initial_prompt_provider: callable returning the current prompt string,
        # or a plain string. Whisper biases its output toward words in this prompt
        # — used to lock in unit names ("E92"), local proper nouns ("Oroville"),
        # call types, and street names so they don't get mistranscribed.
        self.initial_prompt_provider = initial_prompt_provider
        self._model = None  # lazy-loaded on first use (only when remote unavailable)
        self._lock: asyncio.Lock | None = None  # serialize local model calls
        self._remote_failures = 0
        self._remote_total = 0

    def _current_prompt(self) -> str:
        if not self.initial_prompt_provider:
            return ""
        try:
            if callable(self.initial_prompt_provider):
                return (self.initial_prompt_provider() or "").strip()
            return (str(self.initial_prompt_provider) or "").strip()
        except Exception as e:
            log.debug("initial_prompt_provider raised: %s", e)
            return ""

    def _ensure_model(self):
        if self._model is not None:
            return
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # pywhispercpp.Model handles model download from HuggingFace if not cached
        from pywhispercpp.model import Model

        log.info("loading whisper model %s (downloads ~150 MB on first use)…", self.model_name)
        self._model = Model(
            self.model_name,
            models_dir=str(MODELS_DIR),
            n_threads=4,        # use all 4 CM4 cores
            print_progress=False,
            print_realtime=False,
        )
        log.info("whisper model %s loaded", self.model_name)

    async def _preprocess_audio(self, audio_i16: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run ffmpeg filter chain on int16 audio, return processed int16 array."""
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-af", PREPROCESS_FILTER_CHAIN,
            "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(audio_i16.tobytes())
        if proc.returncode != 0:
            log.warning("preprocess failed: %s", stderr.decode("utf-8", errors="replace")[-200:])
            return audio_i16  # fallback to raw
        return np.frombuffer(stdout, dtype=np.int16)

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text. Routes:
        - If server_url set: try remote, fall back to local on error
        - Otherwise: local only
        """
        if len(audio) == 0:
            return ""

        # Optional acoustic preprocessing — runs once regardless of route
        audio_for_whisper = audio
        sr_for_whisper = sample_rate
        if self.preprocess:
            audio_i16 = audio if audio.dtype == np.int16 else (audio * 32767).astype(np.int16)
            try:
                audio_for_whisper = await self._preprocess_audio(audio_i16, sample_rate)
                sr_for_whisper = 16000
            except FileNotFoundError:
                log.warning("ffmpeg not found — skipping preprocessing")
            except Exception as e:
                log.warning("preprocess error %s — using raw audio", e)

        # Try remote first if configured
        text = ""
        if self.server_url:
            self._remote_total += 1
            try:
                text = await self._transcribe_remote(audio_for_whisper, sr_for_whisper)
            except Exception as e:
                self._remote_failures += 1
                log.warning(
                    "remote whisper failed (%s/%s) — falling back to local: %s",
                    self._remote_failures, self._remote_total, e,
                )
                text = ""

        # Local fallback (also the default if no server_url)
        if not text and (not self.server_url):
            if self._lock is None:
                self._lock = asyncio.Lock()
            async with self._lock:
                text = await self._transcribe_local(audio_for_whisper, sr_for_whisper)

        # Drop known Whisper hallucination boilerplate. These show up on near-silent
        # bursts and would otherwise leak into prealert / phrase matching.
        if _is_hallucination(text):
            log.debug("dropping hallucination: %r", text)
            return ""
        return text

    async def _transcribe_remote(self, audio: np.ndarray, sample_rate: int) -> str:
        """POST audio as a WAV upload to the configured remote whisper server."""
        # Build a WAV in memory
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
        buf.seek(0)

        url = f"{self.server_url}/transcribe"
        prompt = self._current_prompt()
        files = {"file": ("audio.wav", buf.getvalue(), "audio/wav")}
        data_form: dict[str, str] = {}
        if prompt:
            data_form["initial_prompt"] = prompt
        async with httpx.AsyncClient(timeout=self.server_timeout_sec) as client:
            r = await client.post(url, files=files, data=data_form or None)
            r.raise_for_status()
            data = r.json()
        text = (data.get("text") or "").strip()
        log.info(
            "remote transcribed %.1fs in %.2fs (model=%s) -> %d chars",
            data.get("duration_sec", 0),
            data.get("inference_sec", 0),
            data.get("model", "?"),
            len(text),
        )
        return text

    async def _transcribe_local(self, audio: np.ndarray, sample_rate: int) -> str:
        # whisper.cpp expects float32 normalized [-1, 1] at 16 kHz
        if audio.dtype == np.int16:
            x = audio.astype(np.float32) / 32768.0
        else:
            x = audio.astype(np.float32)

        if sample_rate != 16000:
            log.warning(
                "sample_rate=%d not 16000 — whisper expects 16k, results may degrade",
                sample_rate,
            )

        loop = asyncio.get_event_loop()
        prompt = self._current_prompt()
        try:
            self._ensure_model()
            # pywhispercpp.Model.transcribe accepts initial_prompt kwarg in recent
            # versions; fall back gracefully if this build doesn't support it.
            def _run():
                if prompt:
                    try:
                        return self._model.transcribe(x, initial_prompt=prompt)
                    except TypeError:
                        return self._model.transcribe(x)
                return self._model.transcribe(x)
            segments = await loop.run_in_executor(None, _run)
        except Exception as e:
            log.error("local transcription failed: %s", e)
            return ""

        text = " ".join(s.text.strip() for s in segments).strip()
        log.info("local transcribed %d sec -> %d chars", len(audio) // 16000, len(text))
        return text
