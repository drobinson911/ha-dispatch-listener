"""Phrase / keyword detection on transcribed text.

Each trigger has:
- `phrase`: the substring to match (case-insensitive)
- `webhook_url`: optional separate webhook to fire when this phrase matches
                 (in addition to the main code-match webhook)

Returns the list of triggers that matched, so the notifier can both
include them in the main payload and fire any per-phrase webhooks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PhraseTrigger:
    phrase: str
    webhook_url: str = ""

    @property
    def normalized(self) -> str:
        return self.phrase.strip().lower()


class PhraseMatcher:
    def __init__(self, triggers: list[dict] | None = None) -> None:
        self.triggers: list[PhraseTrigger] = []
        for t in triggers or []:
            phrase = (t.get("phrase") or "").strip()
            if not phrase:
                continue
            self.triggers.append(
                PhraseTrigger(phrase=phrase, webhook_url=(t.get("webhook_url") or "").strip())
            )

    def find_matches(self, transcript: str) -> list[PhraseTrigger]:
        if not transcript or not self.triggers:
            return []
        # Word-boundary insensitive match — "fire" matches "structure fire"
        # but not "haywire"
        text = transcript.lower()
        out = []
        for t in self.triggers:
            # Accept either bare substring (looser) or word-boundary (tighter).
            # We use bare substring — dispatch transcripts may have weird tokenization.
            if t.normalized in text:
                out.append(t)
        return out
