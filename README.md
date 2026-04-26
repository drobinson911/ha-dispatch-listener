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
- Fires an HA webhook on configured matches → integrate however you want
  (alert lights, PA announcement, mobile push, MariaDB log, etc.)

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
| `buffer_seconds` | `1800` | Reserved for the rolling buffer feature (not yet implemented in v0.1) |
| `snapshot_pre_seconds` | `5` | Reserved (not yet implemented) |
| `snapshot_post_seconds` | `25` | Reserved |
| `snapshot_dir` | `/share/dispatch_listener/captures` | Reserved |
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
  "timestamp": "2026-04-25T22:50:13.402345+00:00",
  "source": "dispatch_listener"
}
```

Wire up an HA automation triggered by the webhook to do whatever you want
on dispatch.

## Status

**v0.1 — Phase 1a:** Audio capture + DTMF decoder + webhook on match.
Designed to coexist with existing dispatch paths (CAD email, etc.) — start
in additive mode, only replace the existing path once you trust accuracy.

### Roadmap

- v0.2: rolling audio buffer + auto-snapshot WAV around detection events
- v0.3: ingress UI for live activity / level meter / recent codes
- v0.4: additional tone systems (two-tone Plectron, 5-tone, single-tone)
- v0.5: optional Whisper-based voice transcription of post-tone audio

## Limitations

- Currently `parec`-based capture; if your HA Yellow has audio quirks, may
  need to fall back to direct ALSA
- Does not handle audio outside the standard DTMF frequency set (697-1633 Hz)
  — extending to other tone systems requires a new decoder module

## License

MIT — see [LICENSE](LICENSE)
