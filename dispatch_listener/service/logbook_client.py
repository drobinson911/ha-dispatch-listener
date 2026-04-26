"""Client for the Station 91 logbook Worker — used to pull a Whisper
`initial_prompt` built from the D1 incident history (units, call types,
street names). Refreshes on a configurable interval so new streets/calls
show up automatically.

The logbook endpoint (`GET /api/calls/bias`) returns:
    {
      "initial_prompt": "CAL FIRE Butte Unit dispatch. ... Units: E92, T92...",
      "units": [...],
      "call_types": [...],
      "streets": [...],
      "counts": {...}
    }
"""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

log = logging.getLogger(__name__)


class LogbookBiasClient:
    """Holds the current Whisper initial_prompt; refreshes from logbook D1."""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        refresh_hours: float = 24.0,
        request_timeout_sec: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.refresh_seconds = max(60.0, float(refresh_hours) * 3600.0)
        self.request_timeout_sec = request_timeout_sec
        self._prompt: str = ""
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def _fetch_once(self) -> str | None:
        url = f"{self.base_url}/api/calls/bias"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_sec) as client:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            log.warning("logbook bias fetch failed (%s): %s", url, e)
            return None
        prompt = (data.get("initial_prompt") or "").strip()
        counts = data.get("counts") or {}
        log.info(
            "logbook bias refreshed: %d units, %d call_types, %d streets, %d prompt chars",
            counts.get("units", 0),
            counts.get("call_types", 0),
            counts.get("streets", 0),
            counts.get("prompt_chars", len(prompt)),
        )
        return prompt

    async def refresh(self, force: bool = False) -> None:
        if not self.configured:
            return
        async with self._lock:
            now = time.time()
            if not force and self._prompt and (now - self._last_refresh) < self.refresh_seconds:
                return
            new_prompt = await self._fetch_once()
            if new_prompt is not None:
                self._prompt = new_prompt
                self._last_refresh = now

    async def start_periodic_refresh(self) -> None:
        """Fire off a background task that refreshes on the configured interval."""
        if not self.configured or self._task is not None:
            return
        await self.refresh(force=True)

        async def loop():
            while True:
                try:
                    await asyncio.sleep(self.refresh_seconds)
                    await self.refresh(force=True)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning("periodic bias refresh error: %s", e)

        self._task = asyncio.create_task(loop())
