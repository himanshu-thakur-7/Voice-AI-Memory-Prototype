"""OpenAI realtime transcription STT — the accessible default-of-record.

Reuses the OpenAI key you already need for the LLM, so there is NO separate STT vendor or
gated signup. It's true streaming with server-side VAD endpointing over a WebSocket.

Audio choice: the GA realtime API moved the audio format under a nested
``session.audio.input.format`` object, and the g711/µ-law format string has been a moving
target during that migration (type errors reported in the wild). PCM16 @ 24 kHz is the
reliably-accepted format, so we decode Twilio's µ-law 8 kHz → PCM16 → resample to 24 kHz
and send ``audio/pcm``. Resampling is single-digit-ms (docs/latency-budget.md), so this
costs nothing meaningful and removes a class of "it won't connect" failures.

Models: ``gpt-4o-transcribe`` (default, high accuracy), ``gpt-4o-mini-transcribe`` (cheaper),
or ``gpt-realtime-whisper`` (controllable-latency streaming). Configure via OPENAI_TRANSCRIBE_MODEL.

Refs: https://developers.openai.com/api/docs/guides/realtime-transcription
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from app.logging import get_logger
from app.providers.stt_base import STTProvider, Transcript
from app.telephony.audio import resample_pcm16, ulaw_to_pcm16

log = get_logger(__name__)

WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
TARGET_RATE = 24000  # the realtime API's native PCM rate

_DELTA = "conversation.item.input_audio_transcription.delta"
_DONE = "conversation.item.input_audio_transcription.completed"


class OpenAITranscribeSTT(STTProvider):
    name = "openai"

    def __init__(
        self, api_key: str, model: str = "gpt-4o-transcribe", language: str = "en"
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        import websockets  # lazy

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        async with websockets.connect(WS_URL, additional_headers=headers) as ws:
            # GA nested transcription-session config. server_vad lets OpenAI endpoint turns
            # and emit a `.completed` event per utterance (our end-of-turn signal).
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": TARGET_RATE},
                            "transcription": {"model": self._model, "language": self._language},
                            "turn_detection": {"type": "server_vad", "silence_duration_ms": 500},
                        }
                    },
                },
            }))

            async def pump() -> None:
                async for ulaw in frames:
                    pcm24 = resample_pcm16(ulaw_to_pcm16(ulaw), 8000, TARGET_RATE)
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm24).decode(),
                    }))

            pump_task = asyncio.create_task(pump())
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    etype = ev.get("type", "")
                    if etype == _DELTA and ev.get("delta"):
                        yield Transcript(text=ev["delta"], is_final=False)
                    elif etype == _DONE:
                        yield Transcript(text=ev.get("transcript", ""), is_final=True)
                    elif etype == "error":
                        log.error("openai_stt.error", err=ev.get("error"))
            finally:
                pump_task.cancel()
