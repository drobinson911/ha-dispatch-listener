"""Dispatch Listener — entry point.

Reads /data/options.json (HA add-on convention), starts audio capture, runs
tone detection, logs all detected codes, optionally fires a webhook when a
configured match_code is heard.

Phase 1 scope: DTMF detection + stdout logging + optional webhook.
Buffer/snapshot/transcription land in later versions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from capture import PulseCapture, autodetect_source
from detector_dtmf import DTMFDecoder
from notifier import Notifier

OPTIONS_PATH = Path("/data/options.json")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def load_options() -> dict:
    if not OPTIONS_PATH.exists():
        # Allow running outside HA for local dev
        return {
            "pulse_source": "auto",
            "input_gain_db": 0.0,
            "tone_system": "dtmf",
            "match_codes": [],
            "learning_mode": True,
            "webhook_url": "",
            "buffer_seconds": 1800,
            "snapshot_pre_seconds": 5,
            "snapshot_post_seconds": 25,
            "snapshot_dir": "/share/dispatch_listener/captures",
            "log_level": "info",
        }
    return json.loads(OPTIONS_PATH.read_text())


async def main() -> int:
    opts = load_options()

    logging.basicConfig(
        level=getattr(logging, opts.get("log_level", "info").upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log = logging.getLogger("dispatch_listener")
    log.info("starting dispatch_listener v0.1.0")

    pulse_source = opts["pulse_source"]
    if pulse_source == "auto":
        pulse_source = autodetect_source()
        if not pulse_source:
            log.error("no PulseAudio capture source found — set pulse_source explicitly")
            return 1
        log.info("auto-detected pulse source: %s", pulse_source)

    Path(opts["snapshot_dir"]).mkdir(parents=True, exist_ok=True)

    notifier = Notifier(
        webhook_url=opts.get("webhook_url", ""),
        match_codes=set(opts.get("match_codes", [])),
        learning_mode=bool(opts.get("learning_mode", True)),
    )

    decoder = DTMFDecoder(sample_rate=16000)
    capture = PulseCapture(
        source=pulse_source,
        sample_rate=16000,
        chunk_ms=20,
    )

    log.info(
        "config: source=%s tone_system=%s match_codes=%s learning_mode=%s webhook=%s",
        pulse_source,
        opts.get("tone_system"),
        sorted(opts.get("match_codes", [])),
        opts.get("learning_mode"),
        "yes" if opts.get("webhook_url") else "no",
    )

    async for chunk in capture.stream():
        for digit in decoder.feed(chunk):
            log.debug("digit: %s", digit)
        for code in decoder.drain_completed_codes():
            log.info("dispatch code detected: %s", code)
            await notifier.notify(code)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
