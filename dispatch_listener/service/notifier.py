"""Webhook notifier with transcript + phrase-match payload.

Behavior:
- always log the event (info level)
- if learning_mode: log only, no fires
- otherwise: fire main webhook on code match (with transcript + phrase_matches)
             AND fire any per-phrase webhooks for matched phrases

This keeps the addon decoupled from your HA automations — what you do with
the webhook is up to you.
"""
from __future__ import annotations

import datetime as dt
import logging

import httpx

from phrase_matcher import PhraseTrigger

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

    async def notify(
        self,
        code: str,
        transcript: str = "",
        phrase_matches: list[PhraseTrigger] | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        phrase_matches = phrase_matches or []
        is_match = code in self.match_codes
        log.info(
            "event code=%s match=%s learning=%s transcript_chars=%d phrase_matches=%s",
            code,
            "yes" if is_match else "no",
            self.learning_mode,
            len(transcript),
            [t.phrase for t in phrase_matches],
        )

        if self.learning_mode:
            # Log the transcript so you can see what whisper heard, but don't fire
            if transcript:
                log.info("transcript: %s", transcript)
            return

        payload = {
            "code": code,
            "transcript": transcript,
            "phrase_matches": [t.phrase for t in phrase_matches],
            "snapshot_path": snapshot_path,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "dispatch_listener",
        }

        # Main webhook — fires on configured code match
        if is_match and self.webhook_url:
            await self._post(self.webhook_url, payload, label=f"code={code}")
        elif is_match and not self.webhook_url:
            log.warning("matched code %s but no webhook_url configured", code)

        # Per-phrase webhooks — fire regardless of code match (if configured)
        for trigger in phrase_matches:
            if trigger.webhook_url:
                phrase_payload = {
                    **payload,
                    "matched_phrase": trigger.phrase,
                }
                await self._post(
                    trigger.webhook_url, phrase_payload, label=f"phrase='{trigger.phrase}'"
                )

    @staticmethod
    async def _post(url: str, payload: dict, label: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(url, json=payload)
                if r.status_code >= 400:
                    log.warning("webhook (%s) returned %s: %s", label, r.status_code, r.text[:200])
                else:
                    log.info("webhook fired (%s) status=%s", label, r.status_code)
        except httpx.HTTPError as e:
            log.error("webhook post failed (%s): %s", label, e)
