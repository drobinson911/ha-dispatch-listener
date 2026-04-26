"""Webhook notifier.

On a confirmed dispatch code:
- always log it (info level)
- if learning_mode: log only, no fire (every code, including unknowns)
- if NOT learning_mode: fire the webhook only when the code matches match_codes
"""
from __future__ import annotations

import datetime as dt
import logging

import httpx

log = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self,
        webhook_url: str,
        match_codes: set[str],
        learning_mode: bool,
    ) -> None:
        self.webhook_url = webhook_url.strip()
        self.match_codes = match_codes
        self.learning_mode = learning_mode

    async def notify(self, code: str) -> None:
        is_match = code in self.match_codes
        log.info(
            "code=%s match=%s learning_mode=%s",
            code,
            "yes" if is_match else "no",
            self.learning_mode,
        )

        if self.learning_mode:
            return  # log-only mode
        if not is_match:
            return
        if not self.webhook_url:
            log.warning("matched code %s but no webhook_url configured", code)
            return

        payload = {
            "code": code,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "dispatch_listener",
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(self.webhook_url, json=payload)
                if r.status_code >= 400:
                    log.warning(
                        "webhook returned %s: %s", r.status_code, r.text[:200]
                    )
                else:
                    log.info("webhook fired for code=%s status=%s", code, r.status_code)
        except httpx.HTTPError as e:
            log.error("webhook post failed for code=%s: %s", code, e)
