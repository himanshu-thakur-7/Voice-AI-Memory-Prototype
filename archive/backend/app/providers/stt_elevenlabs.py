"""ElevenLabs Scribe v2 Realtime STT (alternative recognizer).

Notable for this stack: it accepts Twilio's ``ulaw_8000`` directly (no resampling) and
ships built-in VAD endpointing. GA, ~150 ms latency.

Ref: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from app.providers.stt_base import STTProvider, Transcript

WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class ElevenLabsScribeSTT(STTProvider):
    name = "elevenlabs"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        import websockets  # lazy

        async with websockets.connect(
            WS_URL, additional_headers={"xi-api-key": self._api_key}
        ) as ws:
            # Configure: feed Twilio µ-law straight in; let Scribe's VAD endpoint turns.
            await ws.send(json.dumps({"type": "configure", "audio_format": "ulaw_8000",
                                      "commit_strategy": "vad"}))

            async def pump() -> None:
                async for ulaw in frames:
                    await ws.send(json.dumps({
                        "type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(ulaw).decode(),
                        "sample_rate": 8000,
                    }))
                await ws.send(json.dumps({"type": "flush"}))

            pump_task = asyncio.create_task(pump())
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type", "")
                    if mtype == "partial_transcript":
                        yield Transcript(text=msg.get("text", ""), is_final=False)
                    elif mtype in ("committed_transcript", "final_transcript"):
                        yield Transcript(text=msg.get("text", ""), is_final=True)
            finally:
                pump_task.cancel()
