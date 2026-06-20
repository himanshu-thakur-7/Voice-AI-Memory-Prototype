"""RealtimeEngine — OpenAI Realtime API (speech-to-speech) over one WebSocket.

This is the engine to reach for when sub-second voice-to-voice is a hard requirement:
STT, reasoning, TTS, VAD/turn-detection and barge-in all live inside one streaming model,
so you skip the serial latency stacking of a cascade. It accepts Twilio's ``g711_ulaw``
directly in and out.

Same ``VoiceEngine`` contract as the cascade, so ``main.py`` swaps between them on a flag.
Without an OpenAI key it runs a small mock conversation so engine selection stays demoable.

Ref: https://developers.openai.com/api/docs/guides/realtime-conversations
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json

from app.config import Settings
from app.engines.base import CallContext, Speaker, VoiceEngine, iter_queue
from app.logging import get_logger
from app.metrics import TurnTimer
from app.providers.tts_elevenlabs import MockTTS
from app.vad.turn import SPEECH_END, SPEECH_START, EnergyVAD

log = get_logger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
GREETING = "Hi, thanks for calling. How can I help you today?"


class RealtimeEngine(VoiceEngine):
    name = "realtime"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, inbound: asyncio.Queue, speaker: Speaker, ctx: CallContext) -> None:
        if not self._settings.has_openai:
            log.warning("realtime.mock_mode", call=ctx.call_sid, reason="no OPENAI_API_KEY")
            await _run_mock(inbound, speaker, ctx)
            return
        await self._run_realtime(inbound, speaker, ctx)

    async def _run_realtime(
        self, inbound: asyncio.Queue, speaker: Speaker, ctx: CallContext
    ) -> None:
        import websockets  # lazy

        url = f"{REALTIME_URL}?model={self._settings.openai_realtime_model}"
        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        async with websockets.connect(url, additional_headers=headers) as ws:
            # NOTE: GA gpt-realtime-2 moved formats under session.audio.input/output.format;
            # verify against live docs. We use the widely-deployed flat schema here.
            instructions = ctx.system_prompt or self._settings.base_system_prompt
            if ctx.prosody.realtime_instructions:
                instructions = f"{instructions} {ctx.prosody.realtime_instructions}"
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": instructions,
                    "input_audio_format": "g711_ulaw",   # Twilio's codec — passthrough
                    "output_audio_format": "g711_ulaw",
                    "turn_detection": {"type": "server_vad"},  # native endpointing + barge-in
                    "voice": "alloy",
                },
            }))

            async def send_audio() -> None:
                async for ulaw in iter_queue(inbound):
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(ulaw).decode(),
                    }))

            send_task = asyncio.create_task(send_audio())
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    etype = ev.get("type", "")
                    if etype == "response.audio.delta" and ev.get("delta"):
                        await speaker.play(base64.b64decode(ev["delta"]))
                    elif etype == "input_audio_buffer.speech_started":
                        # The model heard the caller start talking → barge-in.
                        await speaker.clear()
                    elif etype == "error":
                        log.error("realtime.error", call=ctx.call_sid, err=ev.get("error"))
            finally:
                send_task.cancel()


async def _run_mock(inbound: asyncio.Queue, speaker: Speaker, ctx: CallContext) -> None:
    """Endpoint with the energy VAD and reply with a canned tone — mirrors the realtime
    turn-taking shape without a key, so `ENGINE=realtime` is demonstrable offline."""
    tts = MockTTS(tone_hz=330)
    vad = EnergyVAD()
    phrases = itertools.cycle(["sure, one moment", "of course, here you go", "happy to help"])

    async def speak(text: str) -> None:
        timer = TurnTimer()
        async for ulaw in tts.synthesize(text, ctx.prosody, timer):
            await speaker.play(ulaw)
        await speaker.mark(f"{ctx.call_sid}:rt")

    await speak(GREETING)
    spoke = False
    async for frame in iter_queue(inbound):
        ev = vad.update(frame)
        if ev == SPEECH_START:
            spoke = True
            if speaker.speaking.is_set():
                await speaker.clear()  # native-style barge-in
        elif ev == SPEECH_END and spoke:
            spoke = False
            await speak(next(phrases))
    log.info("realtime.mock_end", call=ctx.call_sid)
