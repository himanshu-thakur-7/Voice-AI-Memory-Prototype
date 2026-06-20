"""Ringg Parrot STT V1 — OPTIONAL recognizer (access-gated).

Kept as a ready-to-finish adapter for the Ringg demo: a streaming recognizer (~60 ms
typical latency, Hindi/English/code-mixed) with a ``ringglabs`` Python SDK and documented
Pipecat compatibility (built-in VAD events). The agent natively speaking Ringg's own STT
is a nice touch IF you can get access.

⚠️ Production access is gated (playground / sales@ringg.ai) and the exact ``ringglabs``
streaming method names are not public, so the transport calls below are marked TODO.
Because that access isn't generally available, the **default STT is OpenAI realtime
transcription** (reuses your OpenAI key — see stt_openai.py), with Deepgram/ElevenLabs as
alternatives. This file stays as a drop-in for whoever does have Ringg access: ONLY the
transport methods need filling — audio conversion, endpointing, and the Transcript
contract are already done. The factory falls back to the mock if ``RINGG_API_KEY`` is unset.

Ref: https://www.ringg.ai/models/speech-to-text/v1  (SDK: ``pip install ringglabs``)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.logging import get_logger
from app.providers.stt_base import STTProvider, Transcript
from app.telephony.audio import resample_pcm16, ulaw_to_pcm16

log = get_logger(__name__)

# Ringg recommends a 16 kHz+ PCM input; Twilio gives us 8 kHz µ-law.
RINGG_INPUT_RATE = 16000


class RinggParrotSTT(STTProvider):
    name = "ringg"

    def __init__(self, api_key: str, model: str = "parrot-stt-v1") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        client = self._connect()
        out: asyncio.Queue[Transcript | None] = asyncio.Queue()

        async def pump() -> None:
            try:
                async for ulaw in frames:
                    pcm16 = resample_pcm16(ulaw_to_pcm16(ulaw), 8000, RINGG_INPUT_RATE)
                    await self._send_audio(client, pcm16)
            finally:
                await self._finish(client)

        async def receive() -> None:
            # The Ringg/Pipecat event surface delivers interim + final transcripts and
            # VAD end-of-turn. Map final → is_final=True.
            async for ev in self._events(client):
                await out.put(Transcript(text=ev.text, is_final=ev.is_final))
            await out.put(None)

        pump_task = asyncio.create_task(pump())
        recv_task = asyncio.create_task(receive())
        try:
            while True:
                item = await out.get()
                if item is None:
                    return
                yield item
        finally:
            pump_task.cancel()
            recv_task.cancel()

    # ── transport shims to complete once you have Ringg access ────────────────
    # These four methods are the ENTIRE integration surface. Fill them against the
    # ringglabs SDK (or Ringg's realtime WebSocket) and the provider is production-ready.

    def _connect(self):  # noqa: ANN202
        # from ringglabs import RealtimeSTT
        # return RealtimeSTT(api_key=self._api_key, model=self._model, sample_rate=RINGG_INPUT_RATE)
        raise NotImplementedError(
            "Wire ringglabs here (sales@ringg.ai for access). The default STT is OpenAI "
            "realtime transcription (stt_openai.py) — no Ringg access needed."
        )

    async def _send_audio(self, client, pcm16: bytes) -> None:  # noqa: ANN001
        raise NotImplementedError

    async def _finish(self, client) -> None:  # noqa: ANN001
        raise NotImplementedError

    async def _events(self, client):  # noqa: ANN001, ANN202
        raise NotImplementedError
        yield  # pragma: no cover  (marks this an async generator)
