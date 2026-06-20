"""Twilio Media Streams wire protocol (the JSON the WebSocket carries).

Events FROM Twilio:  connected, start, media, dtmf, mark, stop
Events TO   Twilio:  media, mark, clear

All audio rides inside the JSON ``media.payload`` as base64 µ-law — these are TEXT
frames, not binary WebSocket frames. ``streamSid`` is required on every message we
send back and only becomes known after the ``start`` event.

Ref: https://www.twilio.com/docs/voice/media-streams/websocket-messages
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field


@dataclass
class StartInfo:
    stream_sid: str
    call_sid: str
    account_sid: str
    tracks: list[str]
    custom_parameters: dict[str, str] = field(default_factory=dict)
    media_encoding: str = "audio/x-mulaw"
    media_sample_rate: int = 8000


def parse_event(raw: str) -> tuple[str, dict]:
    """Return (event_name, full_message). event is '' for malformed input."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return "", {}
    return msg.get("event", ""), msg


def parse_start(msg: dict) -> StartInfo:
    s = msg.get("start", {})
    fmt = s.get("mediaFormat", {})
    return StartInfo(
        stream_sid=s.get("streamSid", msg.get("streamSid", "")),
        call_sid=s.get("callSid", ""),
        account_sid=s.get("accountSid", ""),
        tracks=s.get("tracks", []),
        custom_parameters=s.get("customParameters", {}) or {},
        media_encoding=fmt.get("encoding", "audio/x-mulaw"),
        media_sample_rate=int(fmt.get("sampleRate", 8000)),
    )


def decode_media(msg: dict) -> bytes:
    """Extract raw µ-law bytes from an inbound 'media' event."""
    payload = msg.get("media", {}).get("payload", "")
    if not payload:
        return b""
    return base64.b64decode(payload)


# ── outbound message builders ────────────────────────────────────────────────


def media_message(stream_sid: str, ulaw_frame: bytes) -> str:
    """Outbound audio. MUST contain only streamSid + media.payload (no track/chunk/
    timestamp), and the payload must be RAW base64 µ-law — never a WAV/RIFF header."""
    return json.dumps(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(ulaw_frame).decode("ascii")},
        }
    )


def mark_message(stream_sid: str, name: str) -> str:
    """A playback bookmark. Twilio echoes back an identical 'mark' once this point in
    the audio has actually been played to the caller — used to know how much of a reply
    the caller heard before a barge-in."""
    return json.dumps({"event": "mark", "streamSid": stream_sid, "mark": {"name": name}})


def clear_message(stream_sid: str) -> str:
    """Barge-in: immediately empties Twilio's buffered outbound audio (stops the bot
    mid-sentence). Pair with mark tracking to reconcile conversation state."""
    return json.dumps({"event": "clear", "streamSid": stream_sid})
