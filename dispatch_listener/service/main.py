"""Dispatch Listener — entry point.

Pipelines (all running concurrently from the same audio stream):

  parec → audio chunks → audio_buffer (always-on rolling buffer)
                       → DTMF decoder (always-on)
                       → on code: handle_code() task
                            (transcribe + archive + main webhook)
                       → VAD (always-on, optional)
                       → on voice burst end: handle_burst() task
                            (transcribe burst + phrase match + phrase webhooks)
                            ← this is the "pre-alert" path: catches dispatcher
                              voice content BEFORE the tones drop
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from audio_buffer import AudioBuffer
from capture import PulseCapture, autodetect_source
from detector_dtmf import DTMFDecoder
from archiver import Archiver
from notifier import Notifier
from phrase_matcher import PhraseMatcher
from transcriber import Transcriber
from vad import VAD, VADConfig

OPTIONS_PATH = Path("/data/options.json")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

CAPTURE_RATE = 16000
VERSION = "0.4.0"


def load_options() -> dict:
    defaults = {
        "pulse_source": "auto",
        "input_gain_db": 0.0,
        "tone_system": "dtmf",
        "match_codes": [],
        "code_pattern": "",
        "learning_mode": True,
        "webhook_url": "",
        "transcribe_after_match": True,
        "transcribe_seconds": 30,
        "whisper_model": "base.en",
        "phrase_triggers": [],
        "audio_preprocess": True,
        "continuous_transcription": False,
        "vad_activation_db": 12.0,
        "vad_min_burst_seconds": 0.5,
        "vad_max_burst_seconds": 60.0,
        "archive_mode": "snapshot_on_match",
        "archive_pre_seconds": 5,
        "archive_post_seconds": 25,
        "snapshot_dir": "/share/dispatch_listener/captures",
        "log_level": "info",
    }
    if OPTIONS_PATH.exists():
        defaults.update(json.loads(OPTIONS_PATH.read_text()))
    return defaults


async def handle_code(
    code: str,
    *,
    audio_buffer: AudioBuffer,
    transcriber: Transcriber | None,
    archiver: Archiver,
    notifier: Notifier,
    phrase_matcher: PhraseMatcher,
    opts: dict,
) -> None:
    log = logging.getLogger("handler.code")
    log.info("handling code=%s — waiting %ds for post-tone audio…", code, opts["transcribe_seconds"])

    try:
        await audio_buffer.wait_for_seconds_after(float(opts["transcribe_seconds"]))
    except Exception as e:
        log.warning("wait_for_seconds_after failed: %s", e)

    pre = float(opts.get("archive_pre_seconds", 5))
    post = float(opts.get("archive_post_seconds", 25))
    snapshot_audio = audio_buffer.snapshot_tail(pre + post)
    transcribe_audio = audio_buffer.snapshot_tail(float(opts["transcribe_seconds"]))

    should_transcribe = bool(opts.get("transcribe_after_match", True)) and transcriber is not None
    if not opts.get("learning_mode", True) and code not in set(opts.get("match_codes", [])):
        should_transcribe = False

    transcript = ""
    if should_transcribe:
        transcript = await transcriber.transcribe(transcribe_audio, CAPTURE_RATE)

    matches = phrase_matcher.find_matches(transcript)
    snapshot_path = archiver.write_snapshot(code, snapshot_audio)
    snapshot_str = str(snapshot_path) if snapshot_path else None

    await notifier.notify(
        code,
        transcript=transcript,
        phrase_matches=matches,
        snapshot_path=snapshot_str,
    )


async def handle_burst(
    burst_audio: np.ndarray,
    *,
    transcriber: Transcriber,
    notifier: Notifier,
    phrase_matcher: PhraseMatcher,
) -> None:
    """Pre-alert path: transcribe a voice burst, fire phrase webhooks on match."""
    log = logging.getLogger("handler.burst")
    seconds = len(burst_audio) / CAPTURE_RATE
    log.info("transcribing voice burst (%.1fs)", seconds)

    transcript = await transcriber.transcribe(burst_audio, CAPTURE_RATE)
    if not transcript:
        return

    log.info("burst transcript: %s", transcript)
    matches = phrase_matcher.find_matches(transcript)
    if matches:
        # Reuse notifier to fire per-phrase webhooks. code="<voice>" indicates
        # this came from the pre-alert path, not a DTMF event.
        await notifier.notify(
            code="<voice>",
            transcript=transcript,
            phrase_matches=matches,
            snapshot_path=None,
        )


async def main() -> int:
    opts = load_options()
    logging.basicConfig(
        level=getattr(logging, opts.get("log_level", "info").upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log = logging.getLogger("dispatch_listener")
    log.info("starting dispatch_listener v%s", VERSION)

    pulse_source = opts["pulse_source"]
    if pulse_source == "auto":
        pulse_source = autodetect_source()
        if not pulse_source:
            log.error("no PulseAudio capture source found — set pulse_source explicitly")
            return 1
        log.info("auto-detected pulse source: %s", pulse_source)

    Path(opts["snapshot_dir"]).mkdir(parents=True, exist_ok=True)

    match_codes = set(opts.get("match_codes", []))

    decoder = DTMFDecoder(sample_rate=CAPTURE_RATE)
    capture = PulseCapture(source=pulse_source, sample_rate=CAPTURE_RATE, chunk_ms=20)

    buffer_seconds = max(
        float(opts.get("archive_pre_seconds", 5)) + float(opts.get("archive_post_seconds", 25)),
        float(opts.get("transcribe_seconds", 30)),
    ) + 5.0
    audio_buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=CAPTURE_RATE)

    # Whisper transcriber — needed by either the post-tone path OR continuous mode
    transcriber: Transcriber | None = None
    needs_transcriber = bool(opts.get("transcribe_after_match", True)) or bool(
        opts.get("continuous_transcription", False)
    )
    if needs_transcriber:
        transcriber = Transcriber(
            model_name=opts.get("whisper_model", "base.en"),
            preprocess=bool(opts.get("audio_preprocess", True)),
        )
        # Preload model if continuous mode — we want fast first-burst response
        if opts.get("continuous_transcription", False):
            log.info("continuous_transcription enabled — preloading whisper model now…")
            await asyncio.get_event_loop().run_in_executor(None, transcriber._ensure_model)

    archiver = Archiver(
        mode=opts.get("archive_mode", "snapshot_on_match"),
        directory=opts["snapshot_dir"],
        sample_rate=CAPTURE_RATE,
        match_codes=match_codes,
    )
    # Start rolling archive task if configured
    await archiver.start_rolling_task(audio_buffer)

    notifier = Notifier(
        webhook_url=opts.get("webhook_url", ""),
        match_codes=match_codes,
        learning_mode=bool(opts.get("learning_mode", True)),
    )

    phrase_matcher = PhraseMatcher(triggers=opts.get("phrase_triggers", []))

    # VAD — only active when continuous_transcription is on
    vad: VAD | None = None
    burst_chunks: list[np.ndarray] = []
    if opts.get("continuous_transcription", False):
        vad_cfg = VADConfig(
            sample_rate=CAPTURE_RATE,
            chunk_ms=20,
            activation_db=float(opts.get("vad_activation_db", 12.0)),
            min_burst_chunks=int(float(opts.get("vad_min_burst_seconds", 0.5)) * 1000 / 20),
            max_burst_seconds=float(opts.get("vad_max_burst_seconds", 60.0)),
        )
        vad = VAD(vad_cfg)
        log.info(
            "VAD enabled: activation=+%sdB above noise floor, min_burst=%.1fs, max_burst=%.0fs",
            vad_cfg.activation_db,
            vad_cfg.min_burst_chunks * vad_cfg.chunk_ms / 1000.0,
            vad_cfg.max_burst_seconds,
        )

    code_pattern = opts.get("code_pattern", "").strip()
    code_re = re.compile(code_pattern) if code_pattern else None
    if code_re:
        log.info("code_pattern active: codes not matching /%s/ will be dropped", code_pattern)

    log.info(
        "config: source=%s match_codes=%s learning=%s transcribe_match=%s "
        "continuous=%s model=%s archive=%s phrases=%d",
        pulse_source,
        sorted(match_codes),
        opts.get("learning_mode"),
        opts.get("transcribe_after_match"),
        opts.get("continuous_transcription"),
        opts.get("whisper_model"),
        opts.get("archive_mode"),
        len(opts.get("phrase_triggers", [])),
    )

    # Main loop
    async for chunk in capture.stream():
        await audio_buffer.add(chunk)

        # DTMF
        decoder.feed(chunk)
        for code in decoder.drain_completed_codes():
            if code_re and not code_re.fullmatch(code):
                log.debug("dropping code %s (does not match pattern)", code)
                continue
            log.info("dispatch code detected: %s", code)
            asyncio.create_task(
                handle_code(
                    code,
                    audio_buffer=audio_buffer,
                    transcriber=transcriber,
                    archiver=archiver,
                    notifier=notifier,
                    phrase_matcher=phrase_matcher,
                    opts=opts,
                )
            )

        # VAD / continuous transcription
        if vad is not None and transcriber is not None:
            event, _rms_db = vad.feed(chunk)
            if vad.in_burst or event == "burst_end":
                burst_chunks.append(chunk)
            if event == "burst_end" and burst_chunks:
                burst_audio = np.concatenate(burst_chunks)
                burst_chunks = []
                asyncio.create_task(
                    handle_burst(
                        burst_audio,
                        transcriber=transcriber,
                        notifier=notifier,
                        phrase_matcher=phrase_matcher,
                    )
                )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
