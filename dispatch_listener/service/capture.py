"""PulseAudio capture via `parec` subprocess.

Yields mono int16 chunks at the requested sample rate. parec handles all
the format conversion (resample, downmix, sample-format) so the Python
side just reads bytes.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess

import numpy as np

log = logging.getLogger(__name__)


def autodetect_source() -> str | None:
    """Pick the first non-monitor input source from `pactl list short sources`.

    Useful default — most HA Yellow installs only have one USB capture device.
    """
    try:
        out = subprocess.check_output(
            ["pactl", "list", "short", "sources"], text=True, timeout=5
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        if "monitor" in name:
            continue
        return name
    return None


class PulseCapture:
    def __init__(
        self,
        source: str,
        sample_rate: int = 16000,
        chunk_ms: int = 20,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.bytes_per_chunk = self.chunk_samples * 2  # int16

    async def stream(self):
        """Async generator yielding numpy int16 mono arrays."""
        proc = await asyncio.create_subprocess_exec(
            "parec",
            f"--device={self.source}",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            "--channels=1",
            "--latency-msec=20",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        log.info("parec started: pid=%s source=%s", proc.pid, self.source)
        try:
            while True:
                buf = await proc.stdout.readexactly(self.bytes_per_chunk)
                yield np.frombuffer(buf, dtype=np.int16)
        except asyncio.IncompleteReadError:
            log.warning("parec stream ended")
        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
