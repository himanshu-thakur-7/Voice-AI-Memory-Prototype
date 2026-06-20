"""Shared test helpers: synthetic µ-law audio + a fake Twilio WebSocket."""

from __future__ import annotations

import base64
import json
import math

from fastapi import WebSocketDisconnect

from app.telephony.audio import FRAME_BYTES, pcm16_to_ulaw

SILENCE_FRAME = b"\xff" * FRAME_BYTES  # µ-law silence


def speech_frame(tone_hz: int = 220, amp: int = 8000, start_sample: int = 0) -> bytes:
    """A single 20 ms µ-law frame of a loud tone (reads as 'speech' to the energy VAD)."""
    pcm = bytearray(FRAME_BYTES * 2)
    for i in range(FRAME_BYTES):
        s = int(amp * math.sin(2 * math.pi * tone_hz * (start_sample + i) / 8000)) & 0xFFFF
        pcm[2 * i] = s & 0xFF
        pcm[2 * i + 1] = (s >> 8) & 0xFF
    return pcm16_to_ulaw(bytes(pcm))


def frames_for(ms: int, *, kind: str) -> list[bytes]:
    n = max(1, ms // 20)
    if kind == "silence":
        return [SILENCE_FRAME] * n
    return [speech_frame(start_sample=i * FRAME_BYTES) for i in range(n)]


def utterance(pre_silence=120, speech=500, post_silence=640) -> list[bytes]:
    """silence → speech → silence: one full turn the VAD will start and then endpoint."""
    return (
        frames_for(pre_silence, kind="silence")
        + frames_for(speech, kind="speech")
        + frames_for(post_silence, kind="silence")
    )


# ── fake Twilio Media Streams WebSocket ──────────────────────────────────────


def inbound_media(payload: bytes) -> str:
    return json.dumps({"event": "media", "media": {"payload": base64.b64encode(payload).decode()}})


def build_script(stream_sid: str, call_sid: str, frames: list[bytes], custom: dict) -> list[str]:
    msgs = [
        json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
        json.dumps({
            "event": "start",
            "start": {
                "streamSid": stream_sid, "callSid": call_sid, "accountSid": "ACtest",
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": custom,
            },
            "streamSid": stream_sid,
        }),
    ]
    msgs += [inbound_media(f) for f in frames]
    msgs.append(json.dumps({"event": "stop", "streamSid": stream_sid,
                            "stop": {"callSid": call_sid, "accountSid": "ACtest"}}))
    return msgs


class FakeTwilioWS:
    """Duck-typed stand-in for a Starlette WebSocket that replays a Twilio script and
    records everything the server sends back."""

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.sent: list[str] = []

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if self._script:
            return self._script.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000) -> None:
        pass

    # convenience views over what the server sent
    def events(self) -> list[str]:
        return [json.loads(s).get("event", "") for s in self.sent]

    def media_payloads(self) -> list[str]:
        return [json.loads(s)["media"]["payload"] for s in self.sent if '"media"' in s]
