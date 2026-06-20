"""LiveKit audio capture — write the caller's microphone track to a WAV.

The post-call acoustic engine needs a clean, **client-only** audio file (not the mixed
agent/caller stream). LiveKit gives us two paths:

  • **Track Egress** (``StartTrackEgressRequest`` → file/S3, managed server-side) — the
    right call in production: scalable, decoupled from the worker, persistent storage.
  • **In-agent capture** via ``rtc.AudioStream`` — subscribe to the caller's mic track and
    write PCM frames straight to a local WAV. Zero infra to run, ideal for the local demo.

We ship the second path here because the prototype runs on a laptop. The Egress API call
is a single drop-in swap — see ``schedule_egress()`` for the shape.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import wave
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class CallerAudioRecorder:
    """Subscribe to the caller's microphone track and accumulate PCM frames to a WAV.

    Lifecycle:
        rec = CallerAudioRecorder()
        rec.attach(room, target_identity=pctx.participant_identity)   # ON connect
        ...
        wav_path = rec.stop_and_save()                                 # ON disconnect
    """

    def __init__(self) -> None:
        self._frames: list[bytes] = []
        self._sample_rate: int = 16000
        self._num_channels: int = 1
        self._task: asyncio.Task | None = None
        self._target_identity: str = ""

    # ── attach to room events (the real wiring lives here) ──────────────────

    def attach(self, room: Any, target_identity: str) -> None:
        """Hook ``track_subscribed`` so we only consume *this* caller's mic."""
        self._target_identity = target_identity
        try:
            from livekit import rtc  # lazy — keeps tests fast
        except ImportError:
            log.warning("audio_capture.livekit_unavailable",
                        reason="livekit-agents not installed; capture disabled")
            return

        @room.on("track_subscribed")
        def _on_subscribed(track, _publication, participant) -> None:  # noqa: ANN001
            # Only consume the target participant's microphone audio.
            if participant.identity != self._target_identity:
                return
            if not isinstance(track, rtc.RemoteAudioTrack):
                return
            if self._task is not None and not self._task.done():
                return  # already capturing
            self._task = asyncio.create_task(self._consume(track))
            log.info("audio_capture.subscribed", identity=participant.identity,
                     track_name=getattr(track, "name", "?"))

    async def _consume(self, track: Any) -> None:
        """Pull PCM frames from rtc.AudioStream and append to the buffer."""
        from livekit import rtc

        stream = rtc.AudioStream(track)
        try:
            async for event in stream:
                # rtc.AudioStream yields ``AudioFrameEvent(frame=AudioFrame(...))``.
                frame = getattr(event, "frame", event)
                # First frame fixes the sample rate / channels for the WAV header.
                self._sample_rate = getattr(frame, "sample_rate", self._sample_rate)
                self._num_channels = getattr(frame, "num_channels", self._num_channels)
                data = getattr(frame, "data", b"")
                # ``data`` is a memoryview of int16 LE — bytes() snapshots it.
                if data:
                    self._frames.append(bytes(data))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — never crash the agent on a stream hiccup
            log.warning("audio_capture.stream_error", err=str(e))

    # ── stop & save ─────────────────────────────────────────────────────────

    def stop_and_save(self, dest_path: str | None = None) -> str | None:
        """Cancel the capture task and write the WAV. Returns the path, or ``None`` if no
        audio was captured (e.g. caller never enabled their mic or LiveKit unavailable)."""
        if self._task is not None:
            self._task.cancel()
        if not self._frames:
            log.info("audio_capture.empty")
            return None

        if dest_path is None:
            fd, dest_path = tempfile.mkstemp(prefix="cognitive_call_", suffix=".wav")
            os.close(fd)
        try:
            with wave.open(dest_path, "wb") as w:
                w.setnchannels(self._num_channels)
                w.setsampwidth(2)               # PCM16
                w.setframerate(self._sample_rate)
                w.writeframes(b"".join(self._frames))
        except OSError as e:
            log.error("audio_capture.write_failed", err=str(e), path=dest_path)
            with contextlib.suppress(OSError):
                os.remove(dest_path)
            return None

        log.info("audio_capture.saved", path=dest_path,
                 sample_rate=self._sample_rate, channels=self._num_channels,
                 bytes=sum(len(f) for f in self._frames))
        return dest_path


# ── Production path: Track Egress (kept as documentation; not wired) ────────


def schedule_egress(  # pragma: no cover — runs against a real LiveKit server
    livekit_url: str, api_key: str, api_secret: str,
    room_name: str, audio_track_id: str, output_filepath: str,
) -> Any:
    """Recommended *production* path: ask the LiveKit server to record the track for us.

    Lives behind a function so it doesn't pull livekit-api into the import graph until
    you actually need it. The current demo writes locally via ``CallerAudioRecorder``.
    """
    from livekit import api

    client = api.LiveKitAPI(livekit_url, api_key, api_secret)
    req = api.StartTrackEgressRequest(
        room_name=room_name,
        track_id=audio_track_id,
        file=api.EncodedFileOutput(filepath=output_filepath),
    )
    return client.egress.start_track_egress(req)
