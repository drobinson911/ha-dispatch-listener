# Dispatch Listener — Home Assistant add-on

Listens to a radio/scanner audio feed (via USB sound card → PulseAudio),
decodes dispatch tones, and fires a Home Assistant webhook when a configured
station code is heard.

Generic — works with any fire / EMS / police dispatch system that uses
tone-based unit alerting. Tested against CAL FIRE Butte (Oroville) DTMF
dispatch but doesn't bake in any agency-specific config.

## What it does

- Captures live audio from a configurable PulseAudio source (typically a USB
  sound card line-in connected to a radio's audio output)
- Decodes DTMF tone codes (Phase 1) — additional tone systems (two-tone
  Plectron, single-tone, 5-tone) planned
- Logs every detected code so you can build up a catalog of which codes
  belong to which units
- **Transcribes the post-tone voice content** with local Whisper (whisper.cpp,
  no cloud, no PyTorch — runs on a Pi)
- **Phrase / keyword detection** on the transcript — fire separate webhooks
  for things like "structure fire", "code 3", specific street names
- **Audio archive** — snapshot WAV around each event, or 30-min rolling buffer
- Fires an HA webhook on configured matches with the full payload
  (code + transcript + phrase matches + snapshot path)

## How it works

```
[radio audio output] → [USB sound card line-in] → [PulseAudio source]
       → [parec subprocess] → [Goertzel-based DTMF decoder]
       → [HA webhook] → [your automations]
```

The add-on does **only** the audio capture + tone decoding + webhook fire.
What happens after the webhook is up to your HA automations — it's
intentionally decoupled so you can mix-and-match (alerts, logging,
notifications, etc.).

## Hardware setup

You need:
1. A USB sound card with a line-in or mic input (a basic Plugable USB Audio
   Adapter works fine; ~$15)
2. An audio cable from a radio's line-out / aux-out / record-out to the USB
   sound card's input
3. A Home Assistant instance (HAOS recommended) where the add-on can run

For radios without a dedicated line out: tap the cable that feeds your PA
amp, before the amp's volume control. A passive RCA Y-splitter usually
works.

## Installation

This add-on is a custom Home Assistant repository.

1. In HA: **Settings → Add-ons → Add-on Store → ⋮ → Repositories**
2. Add: `https://github.com/drobinson911/ha-dispatch-listener`
3. Find "Dispatch Listener" in the store, install it
4. Configure (see below) and start

## Configuration

| Option | Default | Description |
|---|---|---|
| `pulse_source` | `auto` | PulseAudio source name. `auto` picks the first non-monitor input. Run `pactl list short sources` from a shell add-on to see your options. |
| `input_gain_db` | `0` | Software gain trim in dB (placeholder; v0.1 doesn't apply it yet — set hardware gain via `pactl set-source-volume`) |
| `tone_system` | `dtmf` | Detection algorithm. Currently only `dtmf`. |
| `match_codes` | `[]` | List of codes that should fire the webhook. Empty list + learning_mode = catalog mode. |
| `learning_mode` | `true` | When true, every detected code is logged but NO webhook fires. Use this for the first ~week to discover all your local codes safely. |
| `webhook_url` | `""` | Full HA webhook URL (e.g. `http://homeassistant:8123/api/webhook/<id>`). Required when learning_mode is off. |
| `transcribe_after_match` | `true` | Run Whisper on the post-tone audio. In learning_mode, transcribes every code; in production mode, only matched codes (saves CPU). |
| `transcribe_seconds` | `30` | How much post-tone audio to transcribe |
| `whisper_model` | `base.en` | `tiny.en` / `base.en` / `small.en` (or non-`.en` for multilingual) |
| `phrase_triggers` | `[]` | List of `{phrase, webhook_url?}`. Phrases are matched case-insensitive substrings against the transcript. |
| `archive_mode` | `snapshot_on_match` | `off` / `snapshot_on_match` / `snapshot_on_any` / `rolling_30min` |
| `archive_pre_seconds` | `5` | Audio kept *before* the tone in the snapshot |
| `archive_post_seconds` | `25` | Audio kept *after* the tone in the snapshot |
| `snapshot_dir` | `/share/dispatch_listener/captures` | Where snapshot WAVs are written (mapped to your HA `/share`) |
| `log_level` | `info` | `debug` / `info` / `warning` / `error` |

### Recommended first run

1. Set `learning_mode: true`, leave `match_codes` empty.
2. Start the add-on, watch the logs for ~a week.
3. Note which codes correspond to which dispatches (you'll learn them by
   correlating add-on log timestamps with dispatches you actually heard).
4. Once you have your station's codes catalogued, add them to `match_codes`,
   set `learning_mode: false`, and configure `webhook_url`.

## Webhook payload

```json
{
  "code": "3901",
  "transcript": "Station 91 first out and medics, Engine 91 respond Code 3 to 1234 Oroville Dam Blvd cross of Bridge St for a structure fire, smoke showing from a single story residential",
  "phrase_matches": ["structure fire"],
  "snapshot_path": "/share/dispatch_listener/captures/2026-04-25_22-50-13_3901.wav",
  "timestamp": "2026-04-25T22:50:13.402345+00:00",
  "source": "dispatch_listener"
}
```

Wire up an HA automation triggered by the webhook to do whatever you want
on dispatch — color the alert lights by call type, push transcripts to a
mobile app, log to a database, etc.

## Status

**v0.2 — Phase 1+2:** Audio capture + DTMF decoder + Whisper transcription
+ phrase triggers + audio archive + webhook fan-out.

Designed to coexist with existing dispatch paths (CAD email, etc.) — start
in additive mode, only replace the existing path once you trust accuracy.

### Roadmap

- v0.3: live audio streaming endpoint (HTTP/HLS, on-demand)
- v0.4: ingress UI for live activity / level meter / recent codes /
  transcripts
- v0.5: additional tone systems (two-tone Plectron, 5-tone, single-tone)
- v0.6: optional cloud Whisper fallback for higher accuracy

## Limitations

- Currently `parec`-based capture; if your HA Yellow has audio quirks, may
  need to fall back to direct ALSA
- Does not handle audio outside the standard DTMF frequency set (697-1633 Hz)
  — extending to other tone systems requires a new decoder module

## License

MIT — see [LICENSE](LICENSE)
