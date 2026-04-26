"""Audio archiver — writes WAV snapshots and/or rolling files to disk.

`archive_mode` is now treated as a comma-separated list (or a single mode
for backwards compatibility) — multiple modes can run simultaneously:

- `off`                — never archive
- `snapshot_on_match`  — write a snapshot when a code matched in match_codes
- `snapshot_on_any`    — write a snapshot for every detected code
- `rolling_30min`      — rotating buffer: writes 5-min files, keeps last 6

Common combo: `snapshot_on_match,rolling_30min` — discrete files for your
station's dispatches (training data, easy retrieval) PLUS continuous rolling
coverage for everything else (other stations' calls, ambient context).

Snapshots are pre/post the event:
   [pre_seconds before tone] [tone] [post_seconds after tone]

Defaults: pre=25, post=25 — pre is large enough to capture the 3-beep
pre-alert that fires ~10-15s before the actual DTMF tones.

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


def _parse_modes(mode: str) -> set[str]:
    """Accept comma-separated list or single mode. Filter to valid set."""
    if not mode:
        return {"off"}
    parts = {m.strip() for m in mode.split(",") if m.strip()}
    valid = parts & VALID_MODES
    invalid = parts - VALID_MODES
    if invalid:
        log.warning("ignoring unknown archive_mode entries: %s", sorted(invalid))
    if not valid:
        return {"off"}
    if "off" in valid and len(valid) > 1:
        valid.discard("off")  # other modes override "off"
    return valid


class Archiver:
    def __init__(
        self,
        mode: str,
        directory: str,
        sample_rate: int,
        match_codes: set[str],
    ) -> None:
        self.modes = _parse_modes(mode)
        self.dir = Path(directory)
        self.sample_rate = sample_rate
        self.match_codes = match_codes
        if self.modes != {"off"}:
            self.dir.mkdir(parents=True, exist_ok=True)
        log.info("archiver modes: %s", sorted(self.modes))

    def should_snapshot(self, code: str) -> bool:
        if "snapshot_on_any" in self.modes:
            return True
        if "snapshot_on_match" in self.modes:
            return code in self.match_codes
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
        """If rolling_30min is among configured modes, start the background writer."""
        if "rolling_30min" not in self.modes:
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
