"""Pre-alert beep sequence detector.

Listens for a sequence of identical short tones (beeps) at a target frequency.
Designed for fire dispatch pre-alert systems that play 2-3 beeps at the start
of every dispatch as a "wake up" signal.

Default config matches CAL FIRE Butte's pre-alert:
- 1000 Hz tone
- 200 ms beep duration
- 100 ms inter-beep gap
- 3 beeps = new incident
- 2 beeps = update on existing incident

The detector emits a `beep_sequence` event when a completed sequence is
detected, with the count (2 or 3+). Goertzel filter is cheap — runs
continuously without measurable CPU cost.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BeepConfig:
    sample_rate: int = 16000
    chunk_ms: int = 20
    target_freq_hz: float = 1000.0
    freq_tolerance_hz: float = 50.0
    # On-period (a single beep)
    min_beep_ms: float = 80
    max_beep_ms: float = 400
    # Off-period (gap between beeps in a sequence)
    min_gap_ms: float = 30
    max_gap_ms: float = 350
    # Sequence end: this much silence after the last beep finalizes the count
    sequence_close_ms: float = 600
    # Detection thresholds
    min_absolute_amp: float = 1e5
    inband_dominance: float = 5.0  # target freq must be N× the next-strongest


def _goertzel_power(samples: np.ndarray, freq: float, sr: int) -> float:
    n = len(samples)
    k = int(0.5 + n * freq / sr)
    omega = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = float(x) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    return s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2


@dataclass
class BeepDetector:
    cfg: BeepConfig = field(default_factory=BeepConfig)
    # State
    _in_beep: bool = False
    _beep_start_ms: float = 0.0
    _beep_count: int = 0
    _last_beep_end_ms: float = 0.0
    _silence_after_beep_ms: float = 0.0
    _now_ms: float = 0.0
    # Output queue
    _completed: list[int] = field(default_factory=list)

    def feed(self, chunk: np.ndarray) -> None:
        """Process one chunk. Detected sequences land in `drain_completed()`."""
        chunk_ms = 1000.0 * len(chunk) / self.cfg.sample_rate
        self._now_ms += chunk_ms

        is_beep = self._is_beep_chunk(chunk)

        if is_beep:
            if not self._in_beep:
                self._in_beep = True
                self._beep_start_ms = self._now_ms
            # else: continuing inside an existing beep
        else:
            if self._in_beep:
                # Beep just ended
                duration_ms = self._now_ms - self._beep_start_ms
                if self.cfg.min_beep_ms <= duration_ms <= self.cfg.max_beep_ms:
                    # Valid beep
                    if self._beep_count == 0:
                        # First beep — start a sequence
                        self._beep_count = 1
                    else:
                        # Subsequent beep — must follow within max_gap_ms of last
                        gap = self._beep_start_ms - self._last_beep_end_ms
                        if self.cfg.min_gap_ms <= gap <= self.cfg.max_gap_ms:
                            self._beep_count += 1
                        else:
                            # Gap out of spec — reset and treat this as new sequence start
                            self._beep_count = 1
                    self._last_beep_end_ms = self._now_ms
                    self._silence_after_beep_ms = 0.0
                else:
                    # Bad duration — ignore as transient
                    pass
                self._in_beep = False

            # Track silence accumulating after the last beep
            if self._beep_count > 0:
                self._silence_after_beep_ms += chunk_ms
                if self._silence_after_beep_ms >= self.cfg.sequence_close_ms:
                    # Sequence finalized — emit
                    if self._beep_count >= 2:
                        self._completed.append(self._beep_count)
                        log.info(
                            "beep sequence: %d beep%s",
                            self._beep_count,
                            "" if self._beep_count == 1 else "s",
                        )
                    self._beep_count = 0
                    self._silence_after_beep_ms = 0.0

    def drain_completed(self) -> list[int]:
        out = self._completed
        self._completed = []
        return out

    def _is_beep_chunk(self, chunk: np.ndarray) -> bool:
        """Run Goertzel at target freq and adjacent freqs to verify dominance."""
        if len(chunk) == 0:
            return False
        x = chunk.astype(np.float32)
        fc = self.cfg.target_freq_hz
        # Compare target against off-band probes (300 Hz below + 300 Hz above)
        target = _goertzel_power(x, fc, self.cfg.sample_rate)
        off_low = _goertzel_power(x, fc - 300, self.cfg.sample_rate)
        off_high = _goertzel_power(x, fc + 300, self.cfg.sample_rate)
        if target < self.cfg.min_absolute_amp:
            return False
        max_off = max(off_low, off_high, 1e-9)
        return target >= self.cfg.inband_dominance * max_off
