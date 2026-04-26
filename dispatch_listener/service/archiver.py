"""Audio archiver — writes WAV snapshots and/or rolling files to disk.

Modes:
- `off`                — never archive
- `snapshot_on_match`  — write a snapshot only when a code matched in match_codes
- `snapshot_on_any`    — write a snapshot for every detected code
- `rolling_30min`      — rotating buffer: writes 5-min files, keeps last 6 (= 30 min)

Snapshots are pre/post the event:
   [pre_seconds before tone] [tone] [post_seconds after tone]

Filename conventions:
- Triggered:  <YYYY-MM-DD_HH-MM-SS>_<code>.wav
- Rolling:    rolling_<YYYY-MM-DD_HH-MM-SS>.wav  (auto-rotating, oldest deleted)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

VALID_MODES = {"off", "snapshot_on_match", "snapshot_on_any", "rolling_30min"}

ROLLING_FILE_SECONDS = 5 * 60   # 5-min files
ROLLING_FILE_COUNT = 6          # keep 6 = 30 min coverage


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
        # rolling_30min handled separately by start_rolling_task
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

    async def start_rolling_task(self, audio_buffer) -> asyncio.Task | None:
        """If mode=rolling_30min, start a background writer."""
        if self.mode != "rolling_30min":
            return None
        return asyncio.create_task(self._rolling_loop(audio_buffer))

    async def _rolling_loop(self, audio_buffer) -> None:
        log.info("rolling archiver started: %d-min files, keep %d (%d min coverage)",
                 ROLLING_FILE_SECONDS // 60, ROLLING_FILE_COUNT,
                 ROLLING_FILE_SECONDS * ROLLING_FILE_COUNT // 60)
        while True:
            try:
                # wait for the next file's worth of audio to accumulate
                await audio_buffer.wait_for_seconds_after(ROLLING_FILE_SECONDS)
            except Exception as e:
                log.warning("rolling: wait failed: %s", e)
                await asyncio.sleep(1)
                continue

            audio = audio_buffer.snapshot_tail(ROLLING_FILE_SECONDS)
            ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = self.dir / f"rolling_{ts}.wav"
            try:
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.sample_rate)
                    if audio.dtype != np.int16:
                        audio = audio.astype(np.int16)
                    w.writeframes(audio.tobytes())
                log.info("rolling saved: %s", path.name)
            except OSError as e:
                log.error("rolling write failed: %s", e)
                continue

            # prune oldest beyond ROLLING_FILE_COUNT
            try:
                rolling_files = sorted(self.dir.glob("rolling_*.wav"))
                while len(rolling_files) > ROLLING_FILE_COUNT:
                    old = rolling_files.pop(0)
                    old.unlink()
                    log.info("rolling pruned: %s", old.name)
            except OSError as e:
                log.warning("rolling prune failed: %s", e)
