"""Dispatch Listener — entry point.

Pipelines (all running concurrently from the same audio stream):

  parec → audio chunks → audio_buffer (always-on rolling buffer)
                       → DTMF decoder (always-on)
                       → on code: handle_code() task
                            (transcribe + archive + main webhook)
                       → VAD (always-on, optional)
                       → on voice burst end: handle_burst() task
                            (transcribe burst + phrase match + phrase webhooks)
                            ← this is the "pre-alert" path: catches dispatcher
                              voice content BEFORE the tones drop
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

from audio_buffer import AudioBuffer
from beep_detector import BeepDetector, BeepConfig
from capture import PulseCapture, autodetect_source
from detector_dtmf import DTMFDecoder
from archiver import Archiver
from logbook_client import LogbookBiasClient
from notifier import Notifier
from phrase_matcher import PhraseMatcher
from prealert_matcher import PreAlertMatch, PreAlertMatcher
from streamer import StreamServer
from transcriber import Transcriber
from vad import VAD, VADConfig

OPTIONS_PATH = Path("/data/options.json")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

CAPTURE_RATE = 16000
VERSION = "0.8.0"


def load_options() -> dict:
    defaults = {
        "pulse_source": "auto",
        "input_gain_db": 0.0,
        "tone_system": "dtmf",
        "match_codes": [],
        "code_pattern": "",
        "learning_mode": True,
        "notify_all_codes": False,
        "webhook_url": "",
        "webhook_routes": [],
        "db_log_webhook_url": "",
        "transcribe_after_match": True,
        "transcribe_seconds": 30,
        "whisper_model": "base.en",
        "phrase_triggers": [],
        "audio_preprocess": True,
        "beep_detection_enabled": True,
        "beep_freq_hz": 1000.0,
        "beep_freq_tolerance_hz": 50.0,
        "beep_min_ms": 80,
        "beep_max_ms": 400,
        "beep_min_gap_ms": 30,
        "beep_max_gap_ms": 350,
        "beep_pre_alert_webhook_url": "",
        "beep_update_webhook_url": "",
        "whisper_server_url": "",
        "whisper_server_timeout_sec": 30.0,
        "continuous_transcription": False,
        "vad_activation_db": 12.0,
        "vad_min_burst_seconds": 0.5,
        "vad_max_burst_seconds": 60.0,
        "prealert_areas": [],
        "prealert_for_us_phrases": [],
        "prealert_call_types": [],
        "prealert_default_webhook_url": "",
        "prealert_fuzzy_threshold": 85,
        "prealert_streaming_interval_sec": 2.0,
        "whisper_initial_prompt": "",
        "whisper_initial_prompt_url": "",
        "whisper_initial_prompt_token": "",
        "whisper_initial_prompt_refresh_hours": 24,
        "archive_mode": "snapshot_on_match,rolling_30min",
        "archive_pre_seconds": 25,
        "archive_post_seconds": 30,
        "snapshot_dir": "/share/dispatch_listener/captures",
        "stream_enabled": False,
        "stream_port": 8765,
        "stream_bitrate_kbps": 96,
        "stream_secret": "",
        "log_level": "info",
    }
    if OPTIONS_PATH.exists():
        defaults.update(json.loads(OPTIONS_PATH.read_text()))
    return defaults


async def handle_code(
    code: str,
    *,
    audio_buffer: AudioBuffer,
    transcriber: Transcriber | None,
    archiver: Archiver,
    notifier: Notifier,
    phrase_matcher: PhraseMatcher,
    opts: dict,
) -> None:
    log = logging.getLogger("handler.code")
    log.info("handling code=%s — waiting %ds for post-tone audio…", code, opts["transcribe_seconds"])

    try:
        await audio_buffer.wait_for_seconds_after(float(opts["transcribe_seconds"]))
    except Exception as e:
        log.warning("wait_for_seconds_after failed: %s", e)

    pre = float(opts.get("archive_pre_seconds", 5))
    post = float(opts.get("archive_post_seconds", 25))
    snapshot_audio = audio_buffer.snapshot_tail(pre + post)
    transcribe_audio = audio_buffer.snapshot_tail(float(opts["transcribe_seconds"]))

    should_transcribe = bool(opts.get("transcribe_after_match", True)) and transcriber is not None
    if not opts.get("learning_mode", True) and code not in set(opts.get("match_codes", [])):
        should_transcribe = False

    transcript = ""
    if should_transcribe:
        transcript = await transcriber.transcribe(transcribe_audio, CAPTURE_RATE)

    matches = phrase_matcher.find_matches(transcript)
    snapshot_path = archiver.write_snapshot(code, snapshot_audio)
    snapshot_str = str(snapshot_path) if snapshot_path else None

    await notifier.notify(
        code,
        transcript=transcript,
        phrase_matches=matches,
        snapshot_path=snapshot_str,
    )


class PrealertSuppressor:
    """Tracks recent DTMF code detections so pre-alert webhooks can yield to
    the higher-priority QuickCall flow. When a DTMF code fires, anything that
    would otherwise fire a pre-alert within `window_sec` seconds is suppressed
    on the addon side. (HA-side `media_player.media_stop` is the primary
    cutover; this is the belt-and-suspenders.)"""

    def __init__(self, window_sec: float = 5.0) -> None:
        self.window_sec = window_sec
        self._last_dtmf_at: float = 0.0

    def mark_dtmf(self) -> None:
        import time as _t
        self._last_dtmf_at = _t.time()

    def suppressed(self) -> bool:
        if self._last_dtmf_at == 0.0:
            return False
        import time as _t
        return (_t.time() - self._last_dtmf_at) < self.window_sec


async def _fire_prealert_webhook(url: str, payload: dict) -> None:
    """Fire-and-forget POST for pre-alert webhooks. Kept tiny + fast on purpose —
    HA needs to see this within ~1s of the trigger words being spoken."""
    if not url:
        return
    log = logging.getLogger("prealert.webhook")
    try:
        import httpx
        from notifier import _is_discord_webhook
        if _is_discord_webhook(url):
            label = payload.get("call_type", "?").upper()
            for_us = payload.get("matched_for_us", "")
            body = {"content": f"🚨 **PRE-ALERT — {label}** (for-us: `{for_us}`)"}
        else:
            body = payload
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(url, json=body)
            if r.status_code >= 400:
                log.warning("prealert webhook returned %s: %s", r.status_code, r.text[:200])
            else:
                log.info(
                    "prealert webhook fired type=%s status=%s",
                    payload.get("call_type"), r.status_code,
                )
    except Exception as e:
        log.warning("prealert webhook failed: %s", e)


async def _interim_prealert_pass(
    audio: np.ndarray,
    *,
    transcriber: Transcriber,
    matcher: PreAlertMatcher,
    notifier: Notifier,
    fired_set: set[str],
    suppressor: PrealertSuppressor | None = None,
) -> None:
    """Mid-burst transcription pass — runs every N seconds while a voice burst
    is still ongoing. The point is to fire the HA pre-alert webhook the *instant*
    we have enough words to identify both for-us and call-type, instead of
    waiting for the dispatcher to finish talking. fired_set deduplicates so we
    don't re-fire each interim pass."""
    if not matcher.configured or fired_set:
        return
    if suppressor is not None and suppressor.suppressed():
        return  # QuickCall just fired — yield to the priority path
    transcript = await transcriber.transcribe(audio, CAPTURE_RATE)
    if not transcript:
        return
    log = logging.getLogger("prealert.interim")
    log.debug("interim transcript: %s", transcript)
    if suppressor is not None and suppressor.suppressed():
        return  # check again after the awaitable — QuickCall could have fired in the meantime
    await _try_prealert(transcript, matcher=matcher, notifier=notifier, fired_set=fired_set)


async def _try_prealert(
    transcript: str,
    *,
    matcher: PreAlertMatcher,
    notifier: Notifier,
    fired_set: set[str],
) -> PreAlertMatch | None:
    """Run the 2-stage prealert match. Fire ONCE per call_type per burst.
    `fired_set` is mutated to track what's already been fired this burst."""
    if not matcher.configured:
        return None
    match = matcher.match(transcript)
    if not match:
        return None
    if match.call_type in fired_set:
        return match  # already fired for this burst
    fired_set.add(match.call_type)

    log = logging.getLogger("prealert")
    log.info(
        "PRE-ALERT match: call_type=%s for_us=%s phrase=%s conf=%.2f",
        match.call_type, match.matched_for_us, match.matched_call_phrase, match.confidence,
    )
    payload = {
        "call_type": match.call_type,
        "matched_for_us": match.matched_for_us,
        "matched_call_phrase": match.matched_call_phrase,
        "confidence": match.confidence,
        "transcript": transcript,
        "source": "dispatch_listener",
    }
    asyncio.create_task(_fire_prealert_webhook(match.webhook_url, payload))
    asyncio.create_task(notifier.log_event("prealert", payload))
    return match


async def handle_burst(
    burst_audio: np.ndarray,
    *,
    transcriber: Transcriber,
    notifier: Notifier,
    phrase_matcher: PhraseMatcher,
    prealert_matcher: PreAlertMatcher | None = None,
    fired_call_types: set[str] | None = None,
    suppressor: PrealertSuppressor | None = None,
) -> None:
    """Pre-alert path: transcribe a voice burst, fire phrase + prealert webhooks on match."""
    log = logging.getLogger("handler.burst")
    seconds = len(burst_audio) / CAPTURE_RATE
    log.info("transcribing voice burst (%.1fs)", seconds)

    transcript = await transcriber.transcribe(burst_audio, CAPTURE_RATE)
    if not transcript:
        return

    log.info("burst transcript: %s", transcript)

    # 2-stage pre-alert match (FAST path) — fires before generic phrase webhooks.
    # fired_call_types is a per-burst dedupe set so streaming transcription doesn't
    # re-fire the same category every 2 seconds.
    if prealert_matcher is not None and (suppressor is None or not suppressor.suppressed()):
        await _try_prealert(
            transcript,
            matcher=prealert_matcher,
            notifier=notifier,
            fired_set=fired_call_types if fired_call_types is not None else set(),
        )

    matches = phrase_matcher.find_matches(transcript)
    if matches:
        # Reuse notifier to fire per-phrase webhooks. code="<voice>" indicates
        # this came from the pre-alert path, not a DTMF event.
        await notifier.notify(
            code="<voice>",
            transcript=transcript,
            phrase_matches=matches,
            snapshot_path=None,
        )


async def main() -> int:
    opts = load_options()
    logging.basicConfig(
        level=getattr(logging, opts.get("log_level", "info").upper(), logging.INFO),
        format=LOG_FORMAT,
        stream=sys.stdout,
    )
    log = logging.getLogger("dispatch_listener")
    log.info("starting dispatch_listener v%s", VERSION)

    pulse_source = opts["pulse_source"]
    if pulse_source == "auto":
        pulse_source = autodetect_source()
        if not pulse_source:
            log.error("no PulseAudio capture source found — set pulse_source explicitly")
            return 1
        log.info("auto-detected pulse source: %s", pulse_source)

    Path(opts["snapshot_dir"]).mkdir(parents=True, exist_ok=True)

    match_codes = set(opts.get("match_codes", []))

    decoder = DTMFDecoder(sample_rate=CAPTURE_RATE)
    capture = PulseCapture(source=pulse_source, sample_rate=CAPTURE_RATE, chunk_ms=20)

    # Beep detector — opt-in via beep_detection_enabled
    beep_detector: BeepDetector | None = None
    if opts.get("beep_detection_enabled", True):
        beep_cfg = BeepConfig(
            sample_rate=CAPTURE_RATE,
            chunk_ms=20,
            target_freq_hz=float(opts.get("beep_freq_hz", 1000.0)),
            freq_tolerance_hz=float(opts.get("beep_freq_tolerance_hz", 50.0)),
            min_beep_ms=float(opts.get("beep_min_ms", 80)),
            max_beep_ms=float(opts.get("beep_max_ms", 400)),
            min_gap_ms=float(opts.get("beep_min_gap_ms", 30)),
            max_gap_ms=float(opts.get("beep_max_gap_ms", 350)),
        )
        beep_detector = BeepDetector(cfg=beep_cfg)
        log.info(
            "beep detector enabled: %.0f Hz, %d-%d ms beep, %d-%d ms gap",
            beep_cfg.target_freq_hz, beep_cfg.min_beep_ms, beep_cfg.max_beep_ms,
            beep_cfg.min_gap_ms, beep_cfg.max_gap_ms,
        )

    buffer_seconds = max(
        float(opts.get("archive_pre_seconds", 5)) + float(opts.get("archive_post_seconds", 25)),
        float(opts.get("transcribe_seconds", 30)),
    ) + 5.0
    # If rolling archive is on (alone or combined), size buffer for ~5 min
    if "rolling_30min" in (opts.get("archive_mode") or ""):
        buffer_seconds = max(buffer_seconds, 305.0)
    audio_buffer = AudioBuffer(max_seconds=buffer_seconds, sample_rate=CAPTURE_RATE)
    log.info("audio buffer sized for %.0f sec (~%d MB)", buffer_seconds, int(buffer_seconds * CAPTURE_RATE * 2 / 1024 / 1024))

    # Whisper transcriber — needed by post-tone path, continuous mode, OR prealert
    transcriber: Transcriber | None = None
    prealert_configured = (
        bool(opts.get("prealert_for_us_phrases")) or bool(opts.get("prealert_areas"))
    ) and (
        bool(opts.get("prealert_call_types")) or bool(opts.get("prealert_default_webhook_url"))
    )
    needs_transcriber = (
        bool(opts.get("transcribe_after_match", True))
        or bool(opts.get("continuous_transcription", False))
        or prealert_configured
    )
    # Whisper bias prompt: a static string from config, OR pulled from the
    # logbook D1 (units / call types / street names). Logbook source wins if
    # configured — it auto-refreshes as new streets/calls accumulate.
    static_prompt = (opts.get("whisper_initial_prompt") or "").strip()
    bias_client = LogbookBiasClient(
        base_url=opts.get("whisper_initial_prompt_url", "") or "",
        token=opts.get("whisper_initial_prompt_token", "") or "",
        refresh_hours=float(opts.get("whisper_initial_prompt_refresh_hours", 24)),
    )

    def _prompt_provider() -> str:
        # Prefer the live D1-derived prompt; fall back to the static config one.
        return bias_client.prompt or static_prompt

    if needs_transcriber:
        transcriber = Transcriber(
            model_name=opts.get("whisper_model", "base.en"),
            preprocess=bool(opts.get("audio_preprocess", True)),
            server_url=opts.get("whisper_server_url", ""),
            server_timeout_sec=float(opts.get("whisper_server_timeout_sec", 30.0)),
            initial_prompt_provider=_prompt_provider,
        )
        if transcriber.server_url:
            log.info("whisper remote server configured: %s (local fallback ready)", transcriber.server_url)
        if bias_client.configured:
            log.info(
                "whisper initial_prompt source: logbook (%s, refresh every %.1fh)",
                bias_client.base_url, bias_client.refresh_seconds / 3600.0,
            )
            await bias_client.start_periodic_refresh()
        elif static_prompt:
            log.info("whisper initial_prompt source: static config (%d chars)", len(static_prompt))
        # Preload local model if continuous mode + no remote — fast first-burst response
        if opts.get("continuous_transcription", False) and not transcriber.server_url:
            log.info("continuous_transcription enabled (local mode) — preloading whisper model…")
            await asyncio.get_event_loop().run_in_executor(None, transcriber._ensure_model)

    archiver = Archiver(
        mode=opts.get("archive_mode", "snapshot_on_match"),
        directory=opts["snapshot_dir"],
        sample_rate=CAPTURE_RATE,
        match_codes=match_codes,
    )
    # Start rolling archive task if configured
    await archiver.start_rolling_task(audio_buffer)

    # webhook_routes from HA addon options is a list of {code, url} objects;
    # convert to dict for the Notifier
    raw_routes = opts.get("webhook_routes", []) or []
    if isinstance(raw_routes, dict):
        routes_dict = raw_routes
    else:
        routes_dict = {
            entry["code"]: entry["url"]
            for entry in raw_routes
            if isinstance(entry, dict) and entry.get("code") and entry.get("url")
        }

    notifier = Notifier(
        webhook_url=opts.get("webhook_url", ""),
        match_codes=match_codes,
        learning_mode=bool(opts.get("learning_mode", True)),
        notify_all_codes=bool(opts.get("notify_all_codes", False)),
        webhook_routes=routes_dict,
        db_log_webhook_url=opts.get("db_log_webhook_url", ""),
    )

    phrase_matcher = PhraseMatcher(triggers=opts.get("phrase_triggers", []))

    prealert_matcher = PreAlertMatcher(
        for_us_phrases=opts.get("prealert_for_us_phrases", []) or [],
        areas=opts.get("prealert_areas", []) or [],
        call_types=opts.get("prealert_call_types", []) or [],
        fuzzy_threshold=int(opts.get("prealert_fuzzy_threshold", 85)),
        default_webhook_url=opts.get("prealert_default_webhook_url", "") or "",
    )
    if prealert_matcher.configured:
        log.info(
            "prealert matcher: %d for-us phrases, %d call types, fuzzy=%d",
            len(prealert_matcher.for_us_phrases),
            len(prealert_matcher.call_types),
            prealert_matcher.fuzzy_threshold,
        )

    # Pre-alert suppression: when DTMF (QuickCall) fires, we yield to the
    # higher-priority flow for 5 seconds so a beep doesn't double up with QC.
    prealert_suppressor = PrealertSuppressor(window_sec=5.0)

    # VAD — active when continuous_transcription is on OR prealert is configured.
    # Prealert needs voice bursts to transcribe + match against the trigger lists.
    vad: VAD | None = None
    burst_chunks: list[np.ndarray] = []
    burst_fired_call_types: set[str] = set()
    burst_chunks_since_last_interim: int = 0
    if opts.get("continuous_transcription", False) or prealert_configured:
        vad_cfg = VADConfig(
            sample_rate=CAPTURE_RATE,
            chunk_ms=20,
            activation_db=float(opts.get("vad_activation_db", 12.0)),
            min_burst_chunks=int(float(opts.get("vad_min_burst_seconds", 0.5)) * 1000 / 20),
            max_burst_seconds=float(opts.get("vad_max_burst_seconds", 60.0)),
        )
        vad = VAD(vad_cfg)
        log.info(
            "VAD enabled: activation=+%sdB above noise floor, min_burst=%.1fs, max_burst=%.0fs",
            vad_cfg.activation_db,
            vad_cfg.min_burst_chunks * vad_cfg.chunk_ms / 1000.0,
            vad_cfg.max_burst_seconds,
        )

    code_pattern = opts.get("code_pattern", "").strip()
    code_re = re.compile(code_pattern) if code_pattern else None
    if code_re:
        log.info("code_pattern active: codes not matching /%s/ will be dropped", code_pattern)

    # Live audio streaming (optional, on-demand ffmpeg)
    stream_server: StreamServer | None = None
    if opts.get("stream_enabled", False):
        stream_server = StreamServer(
            pulse_source=pulse_source,
            port=int(opts.get("stream_port", 8765)),
            bitrate_kbps=int(opts.get("stream_bitrate_kbps", 96)),
            secret=opts.get("stream_secret", ""),
        )
        await stream_server.start()
        auth_note = "auth: token required" if stream_server.secret else "auth: NONE (open)"
        log.info(
            "live audio streaming enabled at http://0.0.0.0:%d/stream.mp3 (%d kbps MP3, %s)",
            stream_server.port, stream_server.bitrate_kbps, auth_note,
        )

    log.info(
        "config: source=%s match_codes=%s learning=%s transcribe_match=%s "
        "continuous=%s model=%s archive=%s phrases=%d",
        pulse_source,
        sorted(match_codes),
        opts.get("learning_mode"),
        opts.get("transcribe_after_match"),
        opts.get("continuous_transcription"),
        opts.get("whisper_model"),
        opts.get("archive_mode"),
        len(opts.get("phrase_triggers", [])),
    )

    # Pre-resolved beep webhook URLs
    beep_pre_alert_url = (opts.get("beep_pre_alert_webhook_url") or "").strip()
    beep_update_url = (opts.get("beep_update_webhook_url") or "").strip()

    async def _fire_beep_webhook(url: str, count: int) -> None:
        if not url:
            return
        try:
            import datetime as dt
            import httpx
            from notifier import _is_discord_webhook
            event_name = "pre_alert" if count >= 3 else "incident_update"
            ts = dt.datetime.now(dt.timezone.utc).isoformat()
            if _is_discord_webhook(url):
                emoji = "🚨" if count >= 3 else "🔔"
                label = "PRE-ALERT" if count >= 3 else "INCIDENT UPDATE"
                body = {"content": f"{emoji} **{label}** — {count} beeps detected\n🕒 {ts}"}
            else:
                body = {
                    "event": event_name,
                    "beep_count": count,
                    "timestamp": ts,
                    "source": "dispatch_listener",
                }
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.post(url, json=body)
                if r.status_code >= 400:
                    log.warning("beep webhook returned %s", r.status_code)
        except Exception as e:
            log.warning("beep webhook failed: %s", e)

    # Main loop
    async for chunk in capture.stream():
        await audio_buffer.add(chunk)

        # Beep detector — runs first so pre-alert fires fastest
        if beep_detector is not None:
            beep_detector.feed(chunk)
            for n in beep_detector.drain_completed():
                if n >= 3:
                    log.info("PRE-ALERT detected (%d beeps)", n)
                    asyncio.create_task(_fire_beep_webhook(beep_pre_alert_url, n))
                elif n == 2:
                    log.info("INCIDENT UPDATE detected (%d beeps)", n)
                    asyncio.create_task(_fire_beep_webhook(beep_update_url, n))
                # ALSO log to DB regardless of beep-specific webhook config
                event_type = "pre_alert" if n >= 3 else "incident_update"
                asyncio.create_task(
                    notifier.log_event(event_type, {
                        "beep_count": n,
                        "timestamp": __import__("datetime").datetime.now(
                            __import__("datetime").timezone.utc).isoformat(),
                        "source": "dispatch_listener",
                    })
                )

        # DTMF
        decoder.feed(chunk)
        for code in decoder.drain_completed_codes():
            if code_re and not code_re.fullmatch(code):
                log.debug("dropping code %s (does not match pattern)", code)
                continue
            log.info("dispatch code detected: %s", code)
            # Mark suppression: any in-flight or imminent prealert webhook
            # for the next 5 seconds is yielded to the QuickCall flow.
            prealert_suppressor.mark_dtmf()
            asyncio.create_task(
                handle_code(
                    code,
                    audio_buffer=audio_buffer,
                    transcriber=transcriber,
                    archiver=archiver,
                    notifier=notifier,
                    phrase_matcher=phrase_matcher,
                    opts=opts,
                )
            )

        # VAD / continuous transcription / streaming prealert
        if vad is not None and transcriber is not None:
            event, _rms_db = vad.feed(chunk)
            if event == "burst_start":
                burst_fired_call_types = set()
                burst_chunks_since_last_interim = 0
            if vad.in_burst or event == "burst_end":
                burst_chunks.append(chunk)
                burst_chunks_since_last_interim += 1

            # Streaming/incremental transcription during a long burst:
            # every N seconds while still in_burst, transcribe what we have so far
            # and run prealert match. This lets us fire the HA pre-alert webhook
            # WHILE the dispatcher is still talking, instead of waiting for them
            # to stop speaking.
            if (
                vad.in_burst
                and prealert_matcher.configured
                and burst_chunks
                and len(burst_fired_call_types) == 0  # already fired? no need to keep transcribing
            ):
                interim_sec = float(opts.get("prealert_streaming_interval_sec", 2.0))
                interim_chunks_threshold = max(1, int(interim_sec * 1000 / 20))  # 20ms chunks
                if burst_chunks_since_last_interim >= interim_chunks_threshold:
                    burst_chunks_since_last_interim = 0
                    interim_audio = np.concatenate(burst_chunks)
                    asyncio.create_task(
                        _interim_prealert_pass(
                            interim_audio,
                            transcriber=transcriber,
                            matcher=prealert_matcher,
                            notifier=notifier,
                            fired_set=burst_fired_call_types,
                            suppressor=prealert_suppressor,
                        )
                    )

            if event == "burst_end" and burst_chunks:
                burst_audio = np.concatenate(burst_chunks)
                burst_chunks = []
                # capture the per-burst dedupe set for the final pass + reset for next burst
                fired_set_for_burst = burst_fired_call_types
                burst_fired_call_types = set()
                burst_chunks_since_last_interim = 0
                asyncio.create_task(
                    handle_burst(
                        burst_audio,
                        transcriber=transcriber,
                        notifier=notifier,
                        phrase_matcher=phrase_matcher,
                        prealert_matcher=prealert_matcher if prealert_matcher.configured else None,
                        fired_call_types=fired_set_for_burst,
                        suppressor=prealert_suppressor,
                    )
                )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
