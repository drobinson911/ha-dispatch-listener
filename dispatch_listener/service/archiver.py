"""Audio archiver — writes WAV snapshots to disk.

Modes:
- `off`                — never archive
- `snapshot_on_match`  — write a snapshot only when a code matched in match_codes
- `snapshot_on_any`    — write a snapshot for every detected code
- `rolling_30min`      — rotating buffer of the last 30 min of audio (continuous)

Snapshots are pre/post the event:
   [pre_seconds before tone] [tone] [post_seconds after tone]

Filename convention: <YYYY-MM-DD_HH-MM-SS>_<code>.wav for triggered modes,
                     <YYYY-MM-DD_HH-MM-SS>_rolling.wav for rolling.
"""
from __future__ import annotations

import datetime as dt
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VALID_MODES = {"off", "snapshot_on_match", "snapshot_on_any", "rolling_30min"}


class Archiver:
    def __init__(
        self,
        mode: str,
        directory: str,
        sample_rate: int,
        match_codes: set[str],
    ) -> None:
        if mode not in VALID_MODES:
            log.warning("unknown archive_mode=%s, defaulting to off", mode)
            mode = "off"
        self.mode = mode
        self.dir = Path(directory)
        self.sample_rate = sample_rate
        self.match_codes = match_codes
        if self.mode != "off":
            self.dir.mkdir(parents=True, exist_ok=True)

    def should_snapshot(self, code: str) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "snapshot_on_match":
            return code in self.match_codes
        if self.mode == "snapshot_on_any":
            return True
        # rolling_30min handled separately
        return False

    def write_snapshot(self, code: str, audio: np.ndarray) -> Path | None:
        if not self.should_snapshot(code):
            return None
        if len(audio) == 0:
            return None
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = self.dir / f"{ts}_{code}.wav"
        try:
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                if audio.dtype != np.int16:
                    audio = audio.astype(np.int16)
                w.writeframes(audio.tobytes())
            log.info("snapshot saved: %s (%d sec)", path.name, len(audio) // self.sample_rate)
            return path
        except OSError as e:
            log.error("snapshot write failed: %s", e)
            return None
