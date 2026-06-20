"""Module 3b (part 1) — Affective / paralinguistic extraction.

Runs over the captured caller audio after the call ends and infers an emotional state.
In production the heavy lifting is openSMILE (eGeMAPS) features + a SpeechBrain SER model;
here we ship a dependency-free heuristic over energy / zero-crossing-rate / speech-rate so
the loop produces a plausible AffectiveState offline. Both paths feed the same classifier.
"""

from __future__ import annotations

from app.engines.base import CallContext
from app.logging import get_logger
from app.memory.schemas import AffectiveState, Emotion
from app.telephony.audio import FRAME_BYTES, rms_energy, ulaw_to_pcm16

log = get_logger(__name__)


async def extract_affect(caller_ulaw: bytes, ctx: CallContext) -> AffectiveState:
    if not caller_ulaw:
        return AffectiveState(tenant_id=ctx.tenant_id, user_id=ctx.user_id, confidence=0.0)

    librosa_features = _librosa_features(caller_ulaw)
    features = librosa_features or _heuristic_features(caller_ulaw)
    backend = "librosa" if librosa_features else "heuristic"
    emotion, valence, arousal, confidence = _classify(features)

    log.info("affective.extract", call=ctx.call_sid, user=ctx.user_id,
             emotion=emotion.value, arousal=round(arousal, 2), valence=round(valence, 2),
             backend=backend)

    return AffectiveState(
        tenant_id=ctx.tenant_id, user_id=ctx.user_id, emotion=emotion,
        valence=valence, arousal=arousal, confidence=confidence, features=_strip(features),
    )


def _heuristic_features(ulaw: bytes) -> dict[str, float]:
    """Energy / ZCR / speech-rate features straight from µ-law — no dependencies."""
    energies: list[float] = []
    zcr_total = 0.0
    samples_total = 0
    for off in range(0, len(ulaw), FRAME_BYTES):
        frame = ulaw[off : off + FRAME_BYTES]
        if len(frame) < 8:
            continue
        pcm = ulaw_to_pcm16(frame)
        energies.append(rms_energy(pcm))
        zcr_total += _zero_crossings(pcm)
        samples_total += len(pcm) // 2

    if not energies:
        return {"mean_energy": 0.0, "energy_var": 0.0, "zcr": 0.0, "speech_ratio": 0.0}

    mean_e = sum(energies) / len(energies)
    var_e = sum((e - mean_e) ** 2 for e in energies) / len(energies)
    speech_ratio = sum(1 for e in energies if e > 0.025) / len(energies)
    zcr = (zcr_total / samples_total) if samples_total else 0.0
    return {"mean_energy": mean_e, "energy_var": var_e, "zcr": zcr, "speech_ratio": speech_ratio}


def _librosa_features(ulaw: bytes) -> dict[str, float] | None:
    """Richer features when librosa is installed. Returns None to use the heuristic."""
    try:
        import librosa  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    pcm = ulaw_to_pcm16(ulaw)
    y = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
    if y.size == 0:
        return None
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    var_e = float(np.var(librosa.feature.rms(y=y)))
    tempo = float(librosa.beat.tempo(y=y, sr=8000)[0]) if y.size > 8000 else 0.0
    return {
        "mean_energy": rms, "energy_var": var_e, "zcr": zcr,
        "speech_ratio": float(np.mean(np.abs(y) > 0.02)), "tempo": tempo,
    }
    # Hook for production SER: replace/augment with openSMILE eGeMAPS + a SpeechBrain
    # `foreign_class(... emotion-recognition-wav2vec2-IEMOCAP ...)` classifier.


def _classify(f: dict[str, float]) -> tuple[Emotion, float, float, float]:
    """Map paralinguistic features → (emotion, valence, arousal, confidence).

    Deliberately simple and transparent — a stand-in for a trained SER head. High energy +
    high dynamics + high ZCR reads as activated/negative (frustrated→angry); low energy
    reads as calm/sad; the middle is neutral/positive.
    """
    mean_e = f.get("mean_energy", 0.0)
    var_e = f.get("energy_var", 0.0)
    zcr = f.get("zcr", 0.0)

    arousal = _clamp(mean_e * 6.0 + var_e * 40.0)
    agitation = _clamp(zcr * 4.0 + var_e * 30.0)
    valence = _clamp(0.4 - agitation, lo=-1.0, hi=1.0) if arousal > 0.4 else _clamp(0.2)
    confidence = round(0.4 + 0.3 * min(1.0, f.get("speech_ratio", 0.0) * 2), 2)

    if arousal >= 0.6 and valence < 0:
        emotion = Emotion.ANGRY if arousal >= 0.8 else Emotion.FRUSTRATED
    elif arousal >= 0.5 and agitation >= 0.5:
        emotion = Emotion.ANXIOUS
    elif arousal < 0.25:
        emotion = Emotion.SAD if valence < 0 else Emotion.NEUTRAL
    elif valence > 0.25:
        emotion = Emotion.HAPPY
    else:
        emotion = Emotion.NEUTRAL
    return emotion, round(valence, 3), round(arousal, 3), confidence


def _zero_crossings(pcm16: bytes) -> int:
    prev = 0
    count = 0
    for i in range(0, len(pcm16) - 1, 2):
        s = pcm16[i] | (pcm16[i + 1] << 8)
        if s >= 0x8000:
            s -= 0x10000
        sign = 1 if s >= 0 else -1
        if prev and sign != prev:
            count += 1
        prev = sign
    return count


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _strip(features: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 5) for k, v in features.items() if not k.startswith("_")}
