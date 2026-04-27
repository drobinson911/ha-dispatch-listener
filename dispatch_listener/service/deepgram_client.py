"""Deepgram Nova-3 client for fire dispatch transcription.

Why Deepgram over Whisper for the pre-alert path:
- Trained on telephony/narrowband audio (matches VHF radio bandwidth).
- Keyterm prompting biases the model toward known vocabulary (unit IDs,
  street names, call types) without any training — just paste a list.
- Smart number formatting: "Engine 92" instead of "Engine ninety two".
- Real-time streaming: partial transcripts in ~250ms, final in ~600ms.

This module supports two modes, controlled by `streaming=True/False`:
- Prerecorded: POST a complete WAV to /v1/listen, get a single transcript
  back ~1.5s later. Cheaper, simpler, but waits for burst end.
- Streaming: WebSocket — feed PCM chunks, get interim transcripts as the
  dispatcher speaks. Costs ~80% more, but lets the matcher fire while
  the dispatcher is still talking.

Keyterms come from the same logbook D1 source as the Whisper bias prompt.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import wave
from urllib.parse import quote

import httpx
import numpy as np

log = logging.getLogger(__name__)


def _build_query_params(
    model: str,
    keyterms: list[str],
    extra: dict[str, str] | None = None,
) -> str:
    """Build the ?model=...&keyterm=...&... query string for Deepgram."""
    parts: list[str] = [
        f"model={model}",
        "language=en",
        "smart_format=true",
        "punctuate=true",
        "numerals=true",   # "Engine 92" instead of "Engine ninety two"
    ]
    for kt in keyterms:
        kt = (kt or "").strip()
        if kt:
            parts.append(f"keyterm={quote(kt)}")
    for k, v in (extra or {}).items():
        parts.append(f"{k}={quote(str(v))}")
    return "&".join(parts)


class DeepgramClient:
    """Prerecorded transcription via Deepgram REST API. Streaming variant
    can be added later — for v1 we just POST the full burst once it ends."""

    BASE_URL = "https://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: str,
        model: str = "nova-3",
        request_timeout_sec: float = 15.0,
        keyterm_provider=None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        self.request_timeout_sec = request_timeout_sec
        # keyterm_provider: callable returning current list[str] of bias
        # phrases (units, areas, streets, common call types). Pulled fresh
        # from the logbook D1 client.
        self.keyterm_provider = keyterm_provider

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _current_keyterms(self) -> list[str]:
        if not self.keyterm_provider:
            return []
        try:
            if callable(self.keyterm_provider):
                kts = self.keyterm_provider() or []
            else:
                kts = list(self.keyterm_provider) or []
            return [k for k in kts if k]
        except Exception as e:
            log.debug("keyterm_provider raised: %s", e)
            return []

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Send a WAV-encoded clip to Deepgram, return the transcript text."""
        if not self.configured or len(audio) == 0:
            return ""

        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(audio.tobytes())
        wav_bytes = buf.getvalue()

        keyterms = self._current_keyterms()
        # Deepgram's REST URL has a hard length limit (~8KB) and each
        # keyterm is URL-encoded (~30 chars avg with %20 for spaces). With
        # ~150 keyterms we hit 400 Bad Request. Cap aggressively at 30 —
        # prioritize units + areas + Butte Medics, skip street list
        # (those help less anyway since Deepgram has strong general
        # English street vocabulary, and our matcher fuzzes the rest).
        if len(keyterms) > 30:
            keyterms = keyterms[:30]
        qs = _build_query_params(self.model, keyterms)
        url = f"{self.BASE_URL}?{qs}"
        if len(url) > 7500:
            # Belt-and-suspenders: if still too long, halve until it fits.
            while len(url) > 7500 and len(keyterms) > 5:
                keyterms = keyterms[: max(5, len(keyterms) // 2)]
                qs = _build_query_params(self.model, keyterms)
                url = f"{self.BASE_URL}?{qs}"
            log.warning(
                "deepgram URL near limit, trimmed keyterms to %d (url=%d chars)",
                len(keyterms), len(url),
            )

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout_sec) as client:
                r = await client.post(
                    url,
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": "audio/wav",
                    },
                    content=wav_bytes,
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPError as e:
            log.warning("deepgram request failed: %s", e)
            return ""
        except Exception as e:
            log.warning("deepgram request error: %s", e)
            return ""

        try:
            alt = data["results"]["channels"][0]["alternatives"][0]
            text = (alt.get("transcript") or "").strip()
            confidence = alt.get("confidence", 0.0)
            duration = data.get("metadata", {}).get("duration", 0.0)
            log.info(
                "deepgram transcribed %.1fs audio (model=%s, keyterms=%d, conf=%.2f) -> %d chars",
                duration, self.model, len(keyterms), confidence, len(text),
            )
            return text
        except (KeyError, IndexError, TypeError):
            log.warning("deepgram response missing alternatives: %s", json.dumps(data)[:200])
            return ""
