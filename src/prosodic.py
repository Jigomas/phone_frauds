"""
Prosodic feature scorer.
Extracts speech-rate, silence ratio, and energy variance directly from the WAV
and from Whisper word-level timestamps (when available).

Returns a score in [0, 1]: higher = more likely fraud.
"""

from __future__ import annotations

import math
import numpy as np


def prosodic_score(
    audio_path: str,
    words_with_timestamps: list[dict] | None = None,
) -> float:
    """
    Parameters
    ----------
    audio_path : str
        Path to WAV file.
    words_with_timestamps : list of dicts with keys 'word', 'start', 'end'
        Word-level timestamps from Whisper (optional but improves accuracy).

    Returns
    -------
    float in [0, 1]
    """
    try:
        import librosa
    except ImportError:
        return 0.5  # neutral if librosa unavailable

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return 0.5

    duration = len(y) / sr
    if duration < 1.0:
        return 0.5

    scores: list[float] = []

    # --- 1. Speaking rate (words per second) ---
    if words_with_timestamps:
        n_words = len(words_with_timestamps)
        speech_duration = max(
            words_with_timestamps[-1]["end"] - words_with_timestamps[0]["start"],
            1.0,
        )
        wps = n_words / speech_duration
        # Typical legit: 2-3 wps; fraudsters: 3.5-5 wps (scripted, pressuring)
        # Score ramps from 0 at ≤2.5 wps to 1 at ≥4.5 wps
        rate_score = float(np.clip((wps - 2.5) / 2.0, 0.0, 1.0))
        scores.append(rate_score)

        # --- 2. Silence ratio (pauses > 0.4 s between consecutive words) ---
        pause_total = 0.0
        for i in range(1, len(words_with_timestamps)):
            gap = words_with_timestamps[i]["start"] - words_with_timestamps[i - 1]["end"]
            if gap > 0.4:
                pause_total += gap
        silence_ratio = pause_total / duration
        # Low silence → scripted / pressuring call
        # Legit: 0.15-0.35; fraud: <0.10
        silence_score = float(np.clip(1.0 - (silence_ratio / 0.20), 0.0, 1.0))
        scores.append(silence_score)

    # --- 3. RMS energy variance (emotional pressure) ---
    frame_len = int(sr * 0.025)  # 25 ms frames
    hop_len = int(sr * 0.010)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)
    # High variance → dynamic speech (shouting, urgency)
    rms_std = float(np.std(rms_norm))
    # Typical: legit ~0.25, fraud ~0.35+
    energy_score = float(np.clip((rms_std - 0.20) / 0.20, 0.0, 1.0))
    scores.append(energy_score)

    # --- 4. Zero-crossing rate (voice stress proxy) ---
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr_mean = float(np.mean(zcr))
    # Higher zcr → more high-frequency content / stress in voice
    zcr_score = float(np.clip((zcr_mean - 0.05) / 0.10, 0.0, 1.0))
    scores.append(zcr_score)

    return float(np.mean(scores)) if scores else 0.5
