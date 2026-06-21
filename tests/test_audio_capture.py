"""CallerAudioRecorder — write-WAV path, empty-buffer path, lazy LiveKit import."""

from __future__ import annotations

import os
import wave

from audio_capture import CallerAudioRecorder


def test_empty_buffer_returns_none(tmp_path):
    rec = CallerAudioRecorder()
    out = rec.stop_and_save(str(tmp_path / "out.wav"))
    assert out is None
    assert not (tmp_path / "out.wav").exists()


def test_buffered_frames_write_a_valid_wav(tmp_path):
    rec = CallerAudioRecorder()
    # 1s of silence at 16kHz, mono PCM16 = 32000 bytes.
    rec._sample_rate = 16000
    rec._num_channels = 1
    rec._frames = [b"\x00\x00" * 16000]
    out = rec.stop_and_save(str(tmp_path / "out.wav"))
    assert out is not None
    assert os.path.exists(out)
    # Confirm the header is sane (not a zero-byte placeholder).
    with wave.open(out, "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 16000


def test_attach_no_op_when_livekit_missing():
    """``attach`` must not raise when livekit isn't installed (CI / smoke tests)."""
    rec = CallerAudioRecorder()

    class FakeRoom:
        remote_participants: dict = {}

        def on(self, _event_name):
            def deco(fn):
                return fn
            return deco

    rec.attach(FakeRoom(), target_identity="anyone")    # should not raise
