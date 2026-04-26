"""Webhook notifier with transcript + phrase-match payload.

Behavior:
- always log the event (info level)
- if learning_mode: log only, no fires
- otherwise:
    * Per-code routes (webhook_routes[code]) override the default webhook_url
    * Falls back to webhook_url for matched codes without an explicit route
    * Fires per-phrase webhooks (from phrase_triggers) regardless

Auto-detects Discord webhook URLs (containing `discord.com/api/webhooks`)
and formats payloads as `{"content": "..."}` instead of structured JSON.
"""
from __future__ import annotations

import datetime as dt
import logging

import httpx

from phrase_matcher import PhraseTrigger

log = logging.getLogger(__name__)


def _is_discord_webhook(url: str) -> bool:
    return "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url


def _format_for_discord(payload: dict) -> dict:
    """Convert structured payload into Discord-compatible markdown message."""
    code = payload.get("code", "?")
    matched = payload.get("matched_phrase")
    transcript = (payload.get("transcript") or "").strip()
    phrase_matches = payload.get("phrase_matches") or []
    snapshot = payload.get("snapshot_path")
    ts = payload.get("timestamp", "")

    lines = []
    if code == "<voice>":
        lines.append(f"💬 **Voice phrase match: `{matched or '?'}`**")
    elif matched:
        lines.append(f"💬 **Phrase match: `{matched}`** (during dispatch `{code}`)")
    elif code.startswith("<"):
        lines.append(f"🔔 **Event: {code}**")
    else:
        lines.append(f"🚨 **DISPATCH DETECTED — code `{code}`**")

    if phrase_matches and code != "<voice>":
        lines.append(f"   Phrase matches: {', '.join(f'`{p}`' for p in phrase_matches)}")
    if transcript:
        # Discord limit: 2000 chars per message; transcript usually short
        t = transcript if len(transcript) < 1500 else transcript[:1500] + "…"
        lines.append(f"```\n{t}\n```")
    if snapshot:
        # Show only the basename for tidiness
        from pathlib import Path
        lines.append(f"📼 snapshot: `{Path(snapshot).name}`")
    if ts:
        lines.append(f"🕒 {ts}")

    return {"content": "\n".join(lines)}


class Notifier:
    def __init__(
        self,
        webhook_url: str,
        match_codes: set[str],
        learning_mode: bool,
        webhook_routes: dict[str, str] | None = None,
        db_log_webhook_url: str = "",
    ) -> None:
        self.webhook_url = webhook_url.strip()
        self.match_codes = match_codes
        self.learning_mode = learning_mode
        self.webhook_routes = {
            k.strip(): v.strip()
            for k, v in (webhook_routes or {}).items()
            if k.strip() and v.strip()
        }
        self.db_log_webhook_url = db_log_webhook_url.strip()

    async def log_event(self, event_type: str, payload: dict) -> None:
        """Fire the unified DB-log webhook for any event (code/beep/phrase).
        Independent of webhook_url / webhook_routes / learning_mode — always fires
        if db_log_webhook_url is configured."""
        if not self.db_log_webhook_url:
            return
        log_payload = {**payload, "event_type": event_type}
        await self._post(self.db_log_webhook_url, log_payload, label=f"db_log[{event_type}]")

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

        payload = {
            "code": code,
            "transcript": transcript,
            "phrase_matches": [t.phrase for t in phrase_matches],
            "snapshot_path": snapshot_path,
            "is_match": is_match,
            "learning_mode": self.learning_mode,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "dispatch_listener",
        }

        # ALWAYS log to DB if configured — independent of learning_mode + match
        await self.log_event(
            "phrase_match" if code == "<voice>" else "dtmf_code",
            payload,
        )

        if self.learning_mode:
            if transcript:
                log.info("transcript: %s", transcript)
            # In learning mode, route still fires (Discord monitoring); webhook_url stays silent
            route_url = self.webhook_routes.get(code)
            if route_url:
                await self._post(route_url, payload, label=f"route[{code}]")
            return

        # Production mode: route + webhook_url + per-phrase all fire as appropriate
        # Per-code route takes precedence over default webhook_url for matches
        route_url = self.webhook_routes.get(code)
        if route_url:
            await self._post(route_url, payload, label=f"route[{code}]")
        elif is_match and self.webhook_url:
            await self._post(self.webhook_url, payload, label=f"code={code}")
        elif is_match:
            log.warning("matched code %s but no webhook configured", code)

        # Per-phrase webhooks
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
        # Auto-format for Discord webhooks
        body = _format_for_discord(payload) if _is_discord_webhook(url) else payload
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(url, json=body)
                if r.status_code >= 400:
                    log.warning("webhook (%s) returned %s: %s", label, r.status_code, r.text[:200])
                else:
                    log.info("webhook fired (%s) status=%s", label, r.status_code)
        except httpx.HTTPError as e:
            log.error("webhook post failed (%s): %s", label, e)
