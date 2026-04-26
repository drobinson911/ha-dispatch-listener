"""DTMF decoder using the Goertzel algorithm.

Standard DTMF: each digit is a pair of one low-group + one high-group tone.
Low:  697, 770, 852, 941 Hz
High: 1209, 1336, 1477, 1633 Hz

Detection strategy:
- 20ms input chunks (320 samples @ 16 kHz)
- Run Goertzel for each of the 8 target frequencies on each chunk
- A digit is "present" when the strongest low + strongest high both exceed
  a threshold AND clearly dominate the other 6 frequencies
- A digit is "valid" when it's present for at least MIN_TONE_MS continuous
  milliseconds (filters out transients)
- A "code" is a sequence of 1+ digits separated by no more than MAX_GAP_MS
  of non-detection — closed when MAX_GAP_MS of silence after the last digit

This works for both 4-digit dispatch codes and arbitrary-length sequences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LOW_FREQS = (697, 770, 852, 941)
HIGH_FREQS = (1209, 1336, 1477, 1633)
KEYPAD = (
    ("1", "2", "3", "A"),
    ("4", "5", "6", "B"),
    ("7", "8", "9", "C"),
    ("*", "0", "#", "D"),
)

# Tunables — defaults work for typical dispatch tones (~80ms tone, ~20ms gap)
MIN_TONE_MS = 40        # tone must be present this long to count as a digit
MAX_GAP_MS = 500        # >= this much silence after a digit closes the code
MIN_AMP_RATIO = 4.0     # winning bin must be this much over the median bin
MIN_ABSOLUTE_AMP = 1e6  # absolute floor (squared magnitude scale)


def _goertzel_power(samples: np.ndarray, freq: float, sample_rate: int) -> float:
    """Goertzel single-frequency power estimate."""
    n = len(samples)
    k = int(0.5 + n * freq / sample_rate)
    omega = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = float(x) + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    # power = s_prev^2 + s_prev2^2 - coeff*s_prev*s_prev2
    return s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2


@dataclass
class DTMFDecoder:
    sample_rate: int = 16000
    _current_digit: str | None = None
    _current_digit_ms: float = 0.0
    _building_code: list[str] = field(default_factory=list)
    _silence_ms: float = 0.0
    _completed_codes: list[str] = field(default_factory=list)
    _last_emitted_digit: str | None = None  # avoid re-emitting same digit on long press

    def feed(self, chunk: np.ndarray) -> list[str]:
        """Process one chunk, return any newly-confirmed digits."""
        chunk_ms = 1000.0 * len(chunk) / self.sample_rate
        emitted: list[str] = []

        digit = self._detect_digit(chunk)

        if digit is not None:
            self._silence_ms = 0.0
            if digit == self._current_digit:
                self._current_digit_ms += chunk_ms
            else:
                self._current_digit = digit
                self._current_digit_ms = chunk_ms
                self._last_emitted_digit = None  # new digit, allow emission

            # confirm digit when it's been continuous long enough
            if (
                self._current_digit_ms >= MIN_TONE_MS
                and self._last_emitted_digit != self._current_digit
            ):
                self._building_code.append(self._current_digit)
                emitted.append(self._current_digit)
                self._last_emitted_digit = self._current_digit
        else:
            self._silence_ms += chunk_ms
            self._current_digit = None
            self._current_digit_ms = 0.0
            # close the code if we've been quiet long enough
            if self._building_code and self._silence_ms >= MAX_GAP_MS:
                code = "".join(self._building_code)
                self._completed_codes.append(code)
                self._building_code = []
                self._last_emitted_digit = None

        return emitted

    def drain_completed_codes(self) -> list[str]:
        codes = self._completed_codes
        self._completed_codes = []
        return codes

    def _detect_digit(self, chunk: np.ndarray) -> str | None:
        if len(chunk) == 0:
            return None
        x = chunk.astype(np.float32)
        low_powers = [_goertzel_power(x, f, self.sample_rate) for f in LOW_FREQS]
        high_powers = [_goertzel_power(x, f, self.sample_rate) for f in HIGH_FREQS]

        low_max_i = int(np.argmax(low_powers))
        high_max_i = int(np.argmax(high_powers))
        low_max = low_powers[low_max_i]
        high_max = high_powers[high_max_i]

        # Reject if either band's winner isn't well above the floor
        if low_max < MIN_ABSOLUTE_AMP or high_max < MIN_ABSOLUTE_AMP:
            return None

        # Reject if the winning bin doesn't dominate its band's median
        all_powers = np.array(low_powers + high_powers, dtype=np.float64)
        median = float(np.median(all_powers))
        if median <= 0:
            return None
        if low_max < MIN_AMP_RATIO * median or high_max < MIN_AMP_RATIO * median:
            return None

        return KEYPAD[low_max_i][high_max_i]
