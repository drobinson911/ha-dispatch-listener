"""Whisper transcription via pywhispercpp.

Runs whisper.cpp under the hood — pure C++, no PyTorch, ARM-friendly.
Models cached in /data/models so they persist across addon restarts.

Transcription is sync (whisper.cpp blocks); we run it in an executor so the
asyncio capture loop isn't blocked while we transcribe ~30 sec of audio.

Optional acoustic preprocessing: ffmpeg filter chain that bandpasses to
voice range, denoises, levels loudness BEFORE handing to Whisper. Improves
accuracy on noisy radio audio. ONLY runs on Whisper-bound audio — the
DTMF decoder always reads the raw stream so tone detection is unaffected.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

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


class Transcriber:
    def __init__(self, model_name: str = "base.en", preprocess: bool = True) -> None:
        self.model_name = model_name
        self.preprocess = preprocess
        self._model = None  # lazy-loaded on first use
        # whisper.cpp's Model object is NOT thread/reentrant safe. If two
        # voice bursts overlap (e.g., back-to-back transmissions), parallel
        # calls into model.transcribe() crash with a GGML_ASSERT. Serialize
        # all transcription calls through this lock — bursts queue up.
        self._lock: asyncio.Lock | None = None

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
        """Transcribe a numpy audio array to text. Serialized via lock — overlapping
        bursts queue up to avoid the whisper.cpp non-reentrant crash."""
        if len(audio) == 0:
            return ""

        # Lazy-init the lock so we can construct Transcriber outside an event loop
        if self._lock is None:
            self._lock = asyncio.Lock()

        # Acquire the model lock for the entire preprocess+transcribe critical section.
        # This serializes all whisper calls; bursts queue. Cheap because radio audio
        # is mostly silence anyway.
        async with self._lock:
            return await self._transcribe_locked(audio, sample_rate)

    async def _transcribe_locked(self, audio: np.ndarray, sample_rate: int) -> str:
        # Optional acoustic preprocessing — bandpass + denoise + loudnorm + downsample to 16k
        if self.preprocess:
            audio_i16 = audio if audio.dtype == np.int16 else (audio * 32767).astype(np.int16)
            try:
                processed = await self._preprocess_audio(audio_i16, sample_rate)
                audio = processed
                sample_rate = 16000  # preprocess always outputs 16 kHz for whisper
            except FileNotFoundError:
                log.warning("ffmpeg not found — skipping preprocessing")
            except Exception as e:
                log.warning("preprocess error %s — using raw audio", e)

        # whisper expects float32 normalized [-1, 1] at 16 kHz
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
        try:
            self._ensure_model()
            segments = await loop.run_in_executor(None, self._model.transcribe, x)
        except Exception as e:
            log.error("transcription failed: %s", e)
            return ""

        text = " ".join(s.text.strip() for s in segments).strip()
        log.info("transcribed %d sec → %d chars", len(audio) // 16000, len(text))
        return text
