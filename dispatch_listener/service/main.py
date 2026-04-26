"""Dispatch Listener — entry point.

Pipeline:
  parec → audio chunks → DTMF decoder + audio buffer
                       → on code: spawn handler task that
                           waits for post-event audio,
                           transcribes (Whisper),
                           checks phrase triggers,
                           saves snapshot if configured,
                           fires webhook(s)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from audio_buffer import AudioBuffer
from capture import PulseCapture, autodetect_source
from detector_dtmf import DTMFDecoder
from archiver import Archiver
from notifier import Notifier
from phrase_matcher import PhraseMatcher
from transcriber import Transcriber

OPTIONS_PATH = Path("/data/options.json")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Capture sample rate — fixed at 16 kHz to match Whisper's expectation
CAPTURE_RATE = 16000


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
    """Background task spawned per detected dispatch code."""
    log = logging.getLogger("handler")
    log.info("handling code=%s — waiting %ds for post-tone audio…", code, opts["transcribe_seconds"])

    # Wait for the post-tone audio to accumulate in the buffer
    try:
        await audio_buffer.wait_for_seconds_after(float(opts["transcribe_seconds"]))
    except Exception as e:
        log.warning("wait_for_seconds_after failed: %s", e)

    # Build the slice: pre_seconds + post_seconds (snapshot window)
    pre = float(opts.get("archive_pre_seconds", 5))
    post = float(opts.get("archive_post_seconds", 25))
    snapshot_seconds = pre + post
    snapshot_audio = audio_buffer.snapshot_tail(snapshot_seconds)

    # Transcribe just the post-tone region (skip the pre-buffer which is the tones themselves)
    transcribe_seconds = float(opts["transcribe_seconds"])
    transcribe_audio = audio_buffer.snapshot_tail(transcribe_seconds)

    # Decide whether to transcribe
    should_transcribe = bool(opts.get("transcribe_after_match", True)) and transcriber is not None
    if not opts.get("learning_mode", True) and code not in set(opts.get("match_codes", [])):
        # Not a matched code in production mode → don't burn CPU on transcription
        should_transcribe = False

    transcript = ""
    if should_transcribe:
        transcript = await transcriber.transcribe(transcribe_audio, CAPTURE_RATE)

    # Phrase matching (no-op if no triggers configured or no transcript)
    matches = phrase_matcher.find_matches(transcript)

    # Snapshot if configured
    snapshot_path = archiver.write_snapshot(code, snapshot_audio)
    snapshot_str = str(snapshot_path) if snapshot_path else None

    # Notify
    await notifier.notify(
        code,
        transcript=transcript,
        phrase_matches=matches,
        snapshot_path=snapshot_str,
    )


async def main() -> int:
    opts = load_options()
    logging.basicConfig(
        level=getattr(logging, opts.get("log_level", "info").upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log = logging.getLogger("dispatch_listener")
    log.info("starting dispatch_listener v0.2.0")

    pulse_source = opts["pulse_source"]
    if pulse_source == "auto":
        pulse_source = autodetect_source()
        if not pulse_source:
            log.error("no PulseAudio capture source found — set pulse_source explicitly")
            return 1
        log.info("auto-detected pulse source: %s", pulse_source)

    Path(opts["snapshot_dir"]).mkdir(parents=True, exist_ok=True)

    match_codes = set(opts.get("match_codes", []))

    # Components
    decoder = DTMFDecoder(sample_rate=CAPTURE_RATE)
    capture = PulseCapture(source=pulse_source, sample_rate=CAPTURE_RATE, chunk_ms=20)

    # Buffer needs to hold pre + post + transcribe content with margin
    buffer_seconds = max(
        float(opts.get("archive_pre_seconds", 5)) + float(opts.get("archive_post_seconds", 25)),
        float(opts.get("transcribe_seconds", 30)),
    ) + 5.0  # safety margin
    audio_buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=CAPTURE_RATE)

    # Transcriber lazy-loads model on first use; create only if any path will use it
    transcriber: Transcriber | None = None
    if opts.get("transcribe_after_match", True):
        transcriber = Transcriber(model_name=opts.get("whisper_model", "base.en"))

    archiver = Archiver(
        mode=opts.get("archive_mode", "snapshot_on_match"),
        directory=opts["snapshot_dir"],
        sample_rate=CAPTURE_RATE,
        match_codes=match_codes,
    )

    notifier = Notifier(
        webhook_url=opts.get("webhook_url", ""),
        match_codes=match_codes,
        learning_mode=bool(opts.get("learning_mode", True)),
    )

    phrase_matcher = PhraseMatcher(triggers=opts.get("phrase_triggers", []))

    log.info(
        "config: source=%s match_codes=%s learning=%s transcribe=%s model=%s archive=%s phrases=%d",
        pulse_source,
        sorted(match_codes),
        opts.get("learning_mode"),
        opts.get("transcribe_after_match"),
        opts.get("whisper_model"),
        opts.get("archive_mode"),
        len(opts.get("phrase_triggers", [])),
    )

    # Optional regex filter for codes — drops noise codes silently
    code_pattern = opts.get("code_pattern", "").strip()
    code_re = re.compile(code_pattern) if code_pattern else None
    if code_re:
        log.info("code_pattern active: codes not matching /%s/ will be dropped", code_pattern)

    # Main capture + dispatch loop
    async for chunk in capture.stream():
        await audio_buffer.add(chunk)
        for digit in decoder.feed(chunk):
            log.debug("digit: %s", digit)
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

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
