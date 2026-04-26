"""Ring buffer of audio chunks.

Holds the last `max_seconds` of audio samples. Used by the transcriber and
archiver to grab a slice of recent + post-event audio after a DTMF code is
detected.

Designed for single-producer / multi-consumer: one capture loop appends
chunks, multiple async tasks may read.
"""
from __future__ import annotations

import asyncio
from collections import deque

import numpy as np


class AudioBuffer:
    def __init__(self, max_seconds: float, sample_rate: int) -> None:
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples = 0
        self._cond = asyncio.Condition()

    async def add(self, chunk: np.ndarray) -> None:
        async with self._cond:
            self._chunks.append(chunk)
            self._total_samples += len(chunk)
            while self._total_samples > self.max_samples and self._chunks:
                old = self._chunks.popleft()
                self._total_samples -= len(old)
            self._cond.notify_all()

    def snapshot_tail(self, seconds: float) -> np.ndarray:
        """Return the most recent `seconds` of audio as a contiguous int16 array."""
        n_samples = int(seconds * self.sample_rate)
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        joined = np.concatenate(list(self._chunks))
        if len(joined) > n_samples:
            return joined[-n_samples:]
        return joined

    async def wait_for_seconds_after(self, seconds: float) -> None:
        """Block until `seconds` of new audio has been added."""
        target = self._total_samples + int(seconds * self.sample_rate)
        async with self._cond:
            while self._total_samples < target:
                await self._cond.wait()

    @property
    def fullness_seconds(self) -> float:
        return self._total_samples / self.sample_rate
