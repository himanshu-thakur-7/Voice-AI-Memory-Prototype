"""Deepgram streaming STT (the fully-wired, runnable fallback).

Deepgram accepts Twilio's µ-law 8 kHz **directly** (encoding=mulaw, sample_rate=8000),
so there is no resampling on the inbound leg. It does server-side endpointing and
exposes ``speech_final`` to mark end-of-turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.logging import get_logger
from app.providers.stt_base import STTProvider, Transcript

log = get_logger(__name__)


class DeepgramSTT(STTProvider):
    name = "deepgram"

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        # Lazy import so the package loads without the SDK installed.
        from deepgram import (  # type: ignore
            DeepgramClient,
            LiveOptions,
            LiveTranscriptionEvents,
        )

        out: asyncio.Queue[Transcript | None] = asyncio.Queue()
        dg = DeepgramClient(self._api_key)
        conn = dg.listen.asyncwebsocket.v("1")

        async def on_message(_, result, **__):
            alt = result.channel.alternatives[0]
            text = alt.transcript
            if not text:
                return
            await out.put(Transcript(text=text, is_final=bool(result.speech_final)))

        async def on_close(_, **__):
            await out.put(None)

        conn.on(LiveTranscriptionEvents.Transcript, on_message)
        conn.on(LiveTranscriptionEvents.Close, on_close)

        await conn.start(
            LiveOptions(
                model=self._model,
                encoding="mulaw",      # Twilio's codec — no transcode needed
                sample_rate=8000,
                channels=1,
                punctuate=True,
                interim_results=True,
                endpointing=300,        # ms of trailing silence to finalise a turn
                utterance_end_ms=1000,
            )
        )

        async def pump() -> None:
            try:
                async for frame in frames:
                    await conn.send(frame)
            finally:
                await conn.finish()

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await out.get()
                if item is None:
                    return
                yield item
        finally:
            pump_task.cancel()
