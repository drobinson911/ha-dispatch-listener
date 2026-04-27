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


# Other "Oroville" sub-areas exist that are NOT Station 91's response district
# (North Oroville, West Oroville, East Oroville, etc). The for-us area phrases
# "oroville" and "oroville city" would otherwise match inside those compounds
# via word-boundary regex — "Oroville" is a standalone word inside "North
# Oroville". Reject the hit if the phrase is preceded by a non-our direction.
_AMBIGUOUS_AREA_PHRASES = {"oroville", "oroville city"}
_NON_OUR_DIRECTIONS = {
    "north", "northeast", "northwest",
    "east", "west",
    "southeast", "southwest",
    # NOTE: "south" is OK — "South Oroville" IS ours. That phrase is in the
    # for_us list verbatim and matches before this exclusion check applies.
}


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
    # Per-call-type area restriction. When set, the call type only fires if
    # one of these areas appears in the transcript before the call_type
    # phrase. The matched area also counts as a for-us hit for THIS call
    # type — so "Palermo, structure" fires even if Palermo isn't in the
    # general prealert_areas / for_us_phrases list, while "Palermo, medical"
    # does NOT fire (medical doesn't have only_after_areas configured).
    # Lets us scope structure-and-commercial-structure responses to extended
    # districts (Palermo, Thermalito, Kelly Ridge) without those areas
    # firing the matcher for unrelated call types.
    only_after_areas: list[str] = field(default_factory=list)


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
                    only_after_areas=[
                        a.strip().lower()
                        for a in (c.get("only_after_areas") or [])
                        if a and a.strip()
                    ],
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
        # Whisper transcribes the dispatcher's "TC" as "T.C." or "T C" — the
        # period-stripping below would leave "t c" which doesn't match the
        # word-boundary "tc" phrase. Pre-collapse common spaced-letter
        # acronyms back to their compact form before generic punctuation
        # normalization.
        s = re.sub(r"\bt\.?\s*c\.?\b", "tc", s)
        s = re.sub(r"\bp\.?\s*c\.?\b", "pc", s)  # so "P.C." also normalizes (no false-match)
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _phrase_hit(self, phrase: str, normalized_text: str) -> tuple[bool, float]:
        """Return (hit, confidence). Word-boundary exact → 1.0, fuzzy → score/100.

        Word-boundary matching matters: phrases like "tc" (Traffic Collision in
        radio shorthand) would otherwise match every "tch" / "watch" in the
        transcript via substring. Also prevents "structure" matching
        "structures" / "infrastructure".

        Phrases containing digits skip the fuzzy fallback. Unit IDs like
        "engine 91" / "engine 63" share most characters and partial_ratio
        scores them at 88%+, leading to false matches across stations
        (Engine 63 from Magalia firing the Engine 91 trigger). Whisper
        renders digits deterministically so fuzzy isn't needed here.
        """
        if not phrase:
            return False, 0.0
        # 1. Word-boundary exact match (cheap, common case)
        if re.search(r"\b" + re.escape(phrase) + r"\b", normalized_text):
            return True, 1.0
        # 2. Fuzzy fallback — only for word-only phrases >= 5 chars.
        has_digit = any(c.isdigit() for c in phrase)
        if (
            HAS_RAPIDFUZZ
            and self.fuzzy_threshold > 0
            and len(phrase) >= 5
            and not has_digit
        ):
            score = fuzz.partial_ratio(phrase, normalized_text)
            if score >= self.fuzzy_threshold:
                return True, score / 100.0
        return False, 0.0

    def _phrase_position(self, phrase: str, normalized_text: str) -> int:
        """Return the starting char index of the FIRST word-boundary occurrence
        of phrase in the normalized text, or -1 if not found."""
        if not phrase:
            return -1
        m = re.search(r"\b" + re.escape(phrase) + r"\b", normalized_text)
        return m.start() if m else -1

    def _is_excluded_area(self, phrase: str, normalized_text: str, pos: int) -> bool:
        """For ambiguous area phrases ("oroville", "oroville city") — return
        True if the match is preceded by a non-Station-91 direction word
        (North/East/West Oroville etc are different stations' districts).
        """
        if phrase not in _AMBIGUOUS_AREA_PHRASES or pos < 0:
            return False
        before = normalized_text[:pos].rstrip()
        if not before:
            return False
        words = before.split()
        if not words:
            return False
        return words[-1] in _NON_OUR_DIRECTIONS

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
        if not transcript:
            return None
        text = self._normalize(transcript)
        if not text:
            return None

        # Stage 1: try to find a regular for-us hit (units + global areas).
        # NOTE: we do NOT early-return on miss — call_types with
        # only_after_areas can rescue the match in Stage 2 (e.g. "Palermo,
        # structure" fires even though Palermo isn't in the general for-us
        # list).
        for_us_hit: tuple[str, float] | None = None
        for p in self.for_us_phrases:
            hit, conf = self._phrase_hit(p, text)
            if not hit:
                continue
            pos = self._phrase_position(p, text)
            if self._is_excluded_area(p, text, pos):
                continue
            for_us_hit = (p, conf)
            break

        # Stage 2: what call type? (priority-ordered, first valid hit wins).
        # Note: a call_type with `only_after_areas` can fire even if Stage 1
        # didn't find a regular for-us hit — the area in only_after_areas
        # counts as the for-us hit for THIS call type only. Allows
        # "Palermo, structure" to match without Palermo being in the
        # general for-us list (and without Palermo medicals firing).
        for rule in self.call_types:
            for phrase in rule.phrases:
                hit, conf = self._phrase_hit(phrase, text)
                if not hit:
                    continue
                phrase_pos = self._phrase_position(phrase, text)

                # Determine the area set valid for THIS call type:
                # - only_after_areas (call-type-specific) OVERRIDES self.areas
                # - otherwise fall back to the global areas
                applicable_areas = rule.only_after_areas or self.areas

                # Find the latest applicable area appearing BEFORE phrase_pos.
                # Reject "Oroville" / "Oroville City" hits that are part of a
                # non-our compound area (North Oroville, West Oroville, etc).
                preceding_area: str | None = None
                preceding_area_pos = -1
                for area in applicable_areas:
                    idx = self._phrase_position(area, text)
                    if not (0 <= idx < phrase_pos):
                        continue
                    if self._is_excluded_area(area, text, idx):
                        continue
                    if idx > preceding_area_pos:
                        preceding_area_pos = idx
                        preceding_area = area

                # must_follow_area: require an area-before-phrase
                if rule.must_follow_area and preceding_area is None:
                    continue

                # Negative-after constraint: next ~N words must not contain
                # any not_followed_by. Used to reject "structure fire alarm".
                if rule.not_followed_by:
                    end = phrase_pos + len(phrase)
                    tail = self._next_words_after(text, end, rule.negative_window_words)
                    if any(neg in tail for neg in rule.not_followed_by):
                        continue

                # Determine effective for-us hit:
                # 1. Prefer Stage 1 hit (units are more informative than areas)
                # 2. Fall back to only_after_areas match — this is what lets
                #    Palermo+structure fire without Palermo in the general
                #    for-us list.
                if for_us_hit is not None:
                    effective_for_us = for_us_hit
                elif rule.only_after_areas and preceding_area:
                    effective_for_us = (preceding_area, 1.0)
                else:
                    continue  # no for-us hit and not rescued by only_after_areas

                overall_conf = min(effective_for_us[1], conf)
                return PreAlertMatch(
                    call_type=rule.type,
                    matched_for_us=effective_for_us[0],
                    matched_call_phrase=phrase,
                    webhook_url=rule.webhook_url,
                    confidence=overall_conf,
                )

        # for_us hit but no call type — return generic match if a default is set
        if for_us_hit is not None and self.default_webhook_url:
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
