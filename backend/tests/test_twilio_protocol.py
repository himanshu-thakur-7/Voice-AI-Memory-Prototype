"""Twilio Media Streams protocol round-trips."""

from __future__ import annotations

import base64
import json

from app.telephony.twilio_protocol import (
    clear_message,
    decode_media,
    mark_message,
    media_message,
    parse_event,
    parse_start,
)
from tests.util import build_script, inbound_media


def test_parse_start_captures_streamsid_and_params():
    script = build_script("MZ123", "CA999", [], {"tenant": "acme", "engine": "realtime"})
    event, msg = parse_event(script[1])
    assert event == "start"
    info = parse_start(msg)
    assert info.stream_sid == "MZ123"
    assert info.call_sid == "CA999"
    assert info.custom_parameters["tenant"] == "acme"
    assert info.media_encoding == "audio/x-mulaw"
    assert info.media_sample_rate == 8000


def test_inbound_media_decodes():
    raw = inbound_media(b"\x01\x02\x03")
    _, msg = parse_event(raw)
    assert decode_media(msg) == b"\x01\x02\x03"


def test_outbound_media_is_minimal_and_headerless():
    payload = b"\xff" * 160
    msg = json.loads(media_message("MZ1", payload))
    # Only streamSid + media.payload — no track/chunk/timestamp.
    assert set(msg.keys()) == {"event", "streamSid", "media"}
    assert set(msg["media"].keys()) == {"payload"}
    decoded = base64.b64decode(msg["media"]["payload"])
    assert decoded == payload  # raw µ-law, no WAV/RIFF header
    assert not decoded.startswith(b"RIFF")


def test_clear_and_mark_shapes():
    clear = json.loads(clear_message("MZ1"))
    assert clear == {"event": "clear", "streamSid": "MZ1"}
    mark = json.loads(mark_message("MZ1", "turn:3"))
    assert mark["event"] == "mark" and mark["mark"]["name"] == "turn:3"


def test_parse_event_handles_garbage():
    event, msg = parse_event("not json{")
    assert event == "" and msg == {}
