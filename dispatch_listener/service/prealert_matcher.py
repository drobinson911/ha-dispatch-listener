"""Two-stage pre-alert detector.

A pre-alert is the dispatcher's voice transmission that happens at (or just
before) the DTMF tones drop. We don't need to transcribe the full address —
we just need to know:
  1. Is this for us? (for_us_phrases — area + station resources)
  2. What kind of call? (call_types — each maps to an HA media file via webhook)

If both fire, we trigger a category-specific HA webhook FAST so the station
hears a pre-built audio cue ("STRUCTURE FIRE", "MEDICAL", etc.) often *while*
the dispatcher is still reading the address.

Phase 2 adds rapidfuzz so "I need to be medics" still matches "Butte Medics".
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz  # type: ignore
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


@dataclass
class CallTypeRule:
    type: str                       # human-readable label, e.g. "structure"
    phrases: list[str]              # any of these in the transcript triggers this type
    webhook_url: str = ""           # HA webhook to fire when this type matches
    priority: int = 0               # higher wins on tie (e.g. "commercial structure" > "structure")
    # Positional constraints — used for ambiguous types where context matters.
    # Example: "Structure" must (a) come AFTER an area phrase like "Oroville"
    # so we don't fire on "structure fire alarm" reports from other agencies,
    # and (b) NOT be followed by "alarm" (which downgrades it to a fire alarm,
    # not an actual structure fire).
    must_follow_area: bool = False
    not_followed_by: list[str] = field(default_factory=list)
    negative_window_words: int = 4


@dataclass
class PreAlertMatch:
    call_type: str
    matched_for_us: str             # which for-us phrase hit
    matched_call_phrase: str        # which call-type phrase hit
    webhook_url: str
    confidence: float = 1.0         # 1.0 = exact, <1.0 = fuzzy


class PreAlertMatcher:
    """2-stage matcher: requires for-us hit AND a call-type hit.

    Args:
        for_us_phrases: e.g. ["Oroville", "Engine 92", "Engine 93", "Station 91"]
        call_types: list of CallTypeRule (or dicts shaped like one)
        fuzzy_threshold: 0-100 partial_ratio score required for fuzzy hits.
                         Set to 0 to require exact substring match only.
        default_webhook_url: fires when for-us hits but no call-type matches.
                             Lets HA still wake up the crew with a generic alert.
    """

    def __init__(
        self,
        for_us_phrases: list[str] | None = None,
        call_types: list[dict | CallTypeRule] | None = None,
        fuzzy_threshold: int = 85,
        default_webhook_url: str = "",
        areas: list[str] | None = None,
    ) -> None:
        # `areas` is the subset of for_us_phrases that represent geographic
        # areas (e.g. "Oroville") as opposed to specific resources (e.g.
        # "Engine 92"). Some call types like "Structure" require an area
        # to precede the type word — that constraint is opt-in per call type
        # via must_follow_area.
        self.areas: list[str] = [
            a.strip().lower() for a in (areas or []) if a and a.strip()
        ]
        # for_us_phrases is the full set of triggers that mean "this is for us".
        # If `areas` is given separately, fold them in so we still match the
        # for-us stage when only an area is mentioned.
        merged_for_us = list(for_us_phrases or [])
        merged_for_us.extend(areas or [])
        seen: set[str] = set()
        self.for_us_phrases: list[str] = []
        for p in merged_for_us:
            n = (p or "").strip().lower()
            if n and n not in seen:
                seen.add(n)
                self.for_us_phrases.append(n)

        self.call_types: list[CallTypeRule] = []
        for c in call_types or []:
            if isinstance(c, CallTypeRule):
                self.call_types.append(c)
                continue
            if not isinstance(c, dict):
                continue
            type_name = (c.get("type") or "").strip()
            phrases = [p.strip().lower() for p in (c.get("phrases") or []) if p and p.strip()]
            if not type_name or not phrases:
                continue
            self.call_types.append(
                CallTypeRule(
                    type=type_name,
                    phrases=phrases,
                    webhook_url=(c.get("webhook_url") or "").strip(),
                    priority=int(c.get("priority") or 0),
                    must_follow_area=bool(c.get("must_follow_area", False)),
                    not_followed_by=[
                        p.strip().lower()
                        for p in (c.get("not_followed_by") or [])
                        if p and p.strip()
                    ],
                    negative_window_words=int(c.get("negative_window_words") or 4),
                )
            )
        # Sort call types by priority desc so "commercial structure" beats "structure"
        self.call_types.sort(key=lambda r: -r.priority)
        self.fuzzy_threshold = max(0, min(100, int(fuzzy_threshold)))
        self.default_webhook_url = default_webhook_url.strip()

    # ── matching helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        # Lowercase, collapse whitespace, strip punctuation that radio TTS
        # transcripts sprinkle in. Keep digits + letters + spaces.
        s = text.lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _phrase_hit(self, phrase: str, normalized_text: str) -> tuple[bool, float]:
        """Return (hit, confidence). Exact substring → 1.0, fuzzy → score/100."""
        if not phrase:
            return False, 0.0
        # 1. Exact substring (cheap, common case)
        if phrase in normalized_text:
            return True, 1.0
        # 2. Fuzzy fallback (typo / mistranscription tolerance)
        if HAS_RAPIDFUZZ and self.fuzzy_threshold > 0:
            score = fuzz.partial_ratio(phrase, normalized_text)
            if score >= self.fuzzy_threshold:
                return True, score / 100.0
        return False, 0.0

    def _phrase_position(self, phrase: str, normalized_text: str) -> int:
        """Return the starting char index of the FIRST occurrence of phrase in
        the normalized text, or -1 if not found. Used for ordering constraints."""
        if not phrase:
            return -1
        idx = normalized_text.find(phrase)
        return idx

    def _earliest_area_position(self, normalized_text: str) -> int:
        """Char index of the earliest area-phrase hit, or -1 if none."""
        best = -1
        for a in self.areas:
            idx = normalized_text.find(a)
            if idx >= 0 and (best == -1 or idx < best):
                best = idx
        return best

    def _next_words_after(self, normalized_text: str, end_idx: int, n: int) -> str:
        """Return the next ~n words after end_idx as a single space-joined string."""
        tail = normalized_text[end_idx:].strip()
        words = tail.split()[:n]
        return " ".join(words)

    # ── public API ─────────────────────────────────────────────────────

    def match(self, transcript: str) -> PreAlertMatch | None:
        if not transcript or not self.for_us_phrases:
            return None
        text = self._normalize(transcript)
        if not text:
            return None

        # Stage 1: is this for us?
        for_us_hit: tuple[str, float] | None = None
        for p in self.for_us_phrases:
            hit, conf = self._phrase_hit(p, text)
            if hit:
                for_us_hit = (p, conf)
                break
        if for_us_hit is None:
            return None

        earliest_area_pos = self._earliest_area_position(text) if self.areas else -1

        # Stage 2: what call type? (priority-ordered, first valid hit wins)
        for rule in self.call_types:
            for phrase in rule.phrases:
                hit, conf = self._phrase_hit(phrase, text)
                if not hit:
                    continue

                # Positional constraint: must come AFTER an area phrase.
                # Important for "Structure" / "Commercial Structure" so we don't
                # fire on dispatches for other agencies that happen to contain
                # "structure" without the Oroville/South Oroville/etc. prefix.
                phrase_pos = self._phrase_position(phrase, text)
                if rule.must_follow_area:
                    if earliest_area_pos < 0 or phrase_pos < 0 or phrase_pos <= earliest_area_pos:
                        continue

                # Negative-after constraint: the next ~N words must not contain
                # any of `not_followed_by`. Used to reject "structure fire alarm".
                if rule.not_followed_by and phrase_pos >= 0:
                    end = phrase_pos + len(phrase)
                    tail = self._next_words_after(text, end, rule.negative_window_words)
                    if any(neg in tail for neg in rule.not_followed_by):
                        continue

                overall_conf = min(for_us_hit[1], conf)
                return PreAlertMatch(
                    call_type=rule.type,
                    matched_for_us=for_us_hit[0],
                    matched_call_phrase=phrase,
                    webhook_url=rule.webhook_url,
                    confidence=overall_conf,
                )

        # for_us hit but no call type — return generic match if a default is set
        if self.default_webhook_url:
            return PreAlertMatch(
                call_type="unknown",
                matched_for_us=for_us_hit[0],
                matched_call_phrase="",
                webhook_url=self.default_webhook_url,
                confidence=for_us_hit[1],
            )
        return None

    @property
    def configured(self) -> bool:
        return bool(self.for_us_phrases) and bool(self.call_types or self.default_webhook_url)
