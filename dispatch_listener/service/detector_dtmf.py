"""DTMF decoder using the Goertzel algorithm.

Standard DTMF: each digit is a pair of one low-group + one high-group tone.
Low:  697, 770, 852, 941 Hz
High: 1209, 1336, 1477, 1633 Hz

Detection strategy:
- 20ms input chunks (320 samples @ 16 kHz)
- Run Goertzel for each of the 8 target frequencies on each chunk
- A digit is "present" when:
    1. Strongest low bin's power > absolute floor
    2. Strongest high bin's power > absolute floor
    3. Strongest low bin dominates the OTHER 3 low bins (≥ ratio)
    4. Strongest high bin dominates the OTHER 3 high bins (≥ ratio)
- A digit is "valid" when present continuously for at least MIN_TONE_MS
- A "code" is a sequence of 1+ digits separated by ≤ MAX_GAP_MS of non-detection;
  closed when MAX_GAP_MS of silence after the last digit

The in-band dominance check (step 3 & 4) is what kills voice false positives:
voice content distributes energy broadly, so no single low or high bin
clearly dominates its band. Real DTMF tones are pure → one bin clearly wins.
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

# Tunables — tuned for radio dispatch DTMF (typically 80-200ms tones, clean signal)
MIN_TONE_MS = 60                # tone must be present this long to count as a digit
MAX_GAP_MS = 500                # >= this much silence after a digit closes the code
INTER_DIGIT_SILENCE_MS = 15     # >= this much silence between same-digit tones counts them separately
INBAND_DOMINANCE = 4.0          # winning bin must be >= this much over 2nd-strongest in its band
MIN_ABSOLUTE_AMP = 1e6          # absolute floor (squared magnitude scale)


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
    return s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2


@dataclass
class DTMFDecoder:
    sample_rate: int = 16000
    _current_digit: str | None = None
    _current_digit_ms: float = 0.0
    _building_code: list[str] = field(default_factory=list)
    _silence_ms: float = 0.0
    _completed_codes: list[str] = field(default_factory=list)
    _last_emitted_digit: str | None = None

    def feed(self, chunk: np.ndarray) -> list[str]:
        chunk_ms = 1000.0 * len(chunk) / self.sample_rate
        emitted: list[str] = []

        digit = self._detect_digit(chunk)

        if digit is not None:
            # Any silence resets the "have I emitted current run yet" flag —
            # so consecutive same-digit tones (e.g. "99" in "3992") each get
            # their own emission rather than collapsing into one long "9".
            if self._silence_ms >= INTER_DIGIT_SILENCE_MS:
                self._last_emitted_digit = None
            self._silence_ms = 0.0
            if digit == self._current_digit:
                self._current_digit_ms += chunk_ms
            else:
                self._current_digit = digit
                self._current_digit_ms = chunk_ms
                self._last_emitted_digit = None

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

        # Absolute amplitude floor
        if low_max < MIN_ABSOLUTE_AMP or high_max < MIN_ABSOLUTE_AMP:
            return None

        # In-band dominance: winning bin must be ≥ INBAND_DOMINANCE × second-strongest
        # in its OWN band. Voice fails this; pure tones pass it.
        low_others = sorted([p for i, p in enumerate(low_powers) if i != low_max_i], reverse=True)
        high_others = sorted([p for i, p in enumerate(high_powers) if i != high_max_i], reverse=True)
        # second-strongest in each band (with tiny epsilon to avoid div-by-zero)
        low_second = max(low_others[0], 1e-9)
        high_second = max(high_others[0], 1e-9)

        if low_max < INBAND_DOMINANCE * low_second:
            return None
        if high_max < INBAND_DOMINANCE * high_second:
            return None

        return KEYPAD[low_max_i][high_max_i]
