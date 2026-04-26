"""Voice Activity Detector (energy + spectral-flux based).

Lightweight, no ML — runs on the audio stream and emits voice-burst
boundaries. Designed to be cheap enough to run continuously on a Pi.

Approach:
- Track short-term RMS energy of incoming chunks
- Maintain a rolling noise floor estimate (median of recent quiet chunks)
- A chunk is "active" when its RMS is `activation_db` above the noise floor
- Emit a "burst start" event after `activation_chunks` consecutive active
- Emit a "burst end" event after `release_chunks` consecutive inactive
- Burst length is bounded (`max_burst_seconds`) to avoid runaway transcription
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VADConfig:
    sample_rate: int = 16000
    chunk_ms: int = 20
    activation_db: float = 12.0          # RMS must be this many dB above noise floor
    activation_chunks: int = 6           # ~120 ms of activity to start a burst
    release_chunks: int = 40             # ~800 ms of silence to end a burst
    noise_floor_window_chunks: int = 250 # ~5 sec rolling median for noise floor
    min_burst_chunks: int = 25           # ignore bursts shorter than ~500 ms
    max_burst_seconds: float = 60.0      # cap any single burst at 60 sec


class VAD:
    def __init__(self, cfg: VADConfig | None = None) -> None:
        self.cfg = cfg or VADConfig()
        self._noise_floor_db: float = -60.0
        self._recent_quiet: deque[float] = deque(maxlen=self.cfg.noise_floor_window_chunks)
        self._consec_active: int = 0
        self._consec_inactive: int = 0
        self._in_burst: bool = False
        self._burst_chunks: int = 0

    def feed(self, chunk: np.ndarray) -> tuple[str | None, float]:
        """Process one chunk; return (event, rms_db).

        event ∈ {None, "burst_start", "burst_end"}
        """
        if len(chunk) == 0:
            return None, -100.0

        # Compute RMS in dBFS-ish scale
        x = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x))) + 1e-9
        rms_db = 20.0 * np.log10(rms)

        active = rms_db >= self._noise_floor_db + self.cfg.activation_db

        event: str | None = None

        if active:
            self._consec_active += 1
            self._consec_inactive = 0
            if not self._in_burst and self._consec_active >= self.cfg.activation_chunks:
                self._in_burst = True
                self._burst_chunks = self._consec_active
                event = "burst_start"
            elif self._in_burst:
                self._burst_chunks += 1
                # safety cap
                if (
                    self._burst_chunks * self.cfg.chunk_ms / 1000.0
                    >= self.cfg.max_burst_seconds
                ):
                    self._in_burst = False
                    self._burst_chunks = 0
                    self._consec_active = 0
                    event = "burst_end"
        else:
            self._consec_inactive += 1
            self._consec_active = 0
            # update noise floor only when inactive
            self._recent_quiet.append(rms_db)
            if self._recent_quiet:
                # use median of recent quiet samples; drift slowly
                target = float(np.median(self._recent_quiet))
                self._noise_floor_db = 0.95 * self._noise_floor_db + 0.05 * target
            if self._in_burst and self._consec_inactive >= self.cfg.release_chunks:
                if self._burst_chunks >= self.cfg.min_burst_chunks:
                    event = "burst_end"
                # else: too short, suppress (treat as transient)
                self._in_burst = False
                self._burst_chunks = 0

        return event, rms_db

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    @property
    def in_burst(self) -> bool:
        return self._in_burst

    @property
    def burst_seconds(self) -> float:
        return self._burst_chunks * self.cfg.chunk_ms / 1000.0
