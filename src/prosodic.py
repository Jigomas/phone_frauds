"""
Просодический анализ — оценивает «как» говорит собеседник, а не «что».
Извлекает темп речи, долю тишины и вариацию энергии из WAV-файла.

Возвращает оценку от 0 до 1: чем выше — тем больше признаков мошеннического звонка.
"""

from __future__ import annotations

import math
import numpy as np


def prosodic_score(
    audio_path: str,
    words_with_timestamps: list[dict] | None = None,
) -> float:
    """
    audio_path : путь к WAV-файлу
    words_with_timestamps : слова с временны́ми метками от ASR (опционально, повышает точность)

    Возвращает float от 0 до 1.
    """
    try:
        import librosa
    except ImportError:
        return 0.5  # нет librosa — возвращаем нейтральный счёт

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        return 0.5

    duration = len(y) / sr
    if duration < 1.0:
        return 0.5

    scores: list[float] = []

    # --- 1. Темп речи (слов в секунду) ---
    if words_with_timestamps:
        n_words = len(words_with_timestamps)
        speech_duration = max(
            words_with_timestamps[-1]["end"] - words_with_timestamps[0]["start"],
            1.0,
        )
        wps = n_words / speech_duration
        # Легитимные звонки: обычно 2–3 слова/сек; мошенники говорят быстро — 3.5–5 (скрипт, давление)
        # Счёт 0 при ≤2.5 слова/сек, 1 при ≥4.5
        rate_score = float(np.clip((wps - 2.5) / 2.0, 0.0, 1.0))
        scores.append(rate_score)

        # --- 2. Доля тишины (паузы > 0.4 сек между словами) ---
        pause_total = 0.0
        for i in range(1, len(words_with_timestamps)):
            gap = words_with_timestamps[i]["start"] - words_with_timestamps[i - 1]["end"]
            if gap > 0.4:
                pause_total += gap
        silence_ratio = pause_total / duration
        # Мало тишины → читают по скрипту, торопятся; легитимные: 0.15–0.35, мошенники: <0.10
        silence_score = float(np.clip(1.0 - (silence_ratio / 0.20), 0.0, 1.0))
        scores.append(silence_score)

    # --- 3. Вариация громкости (эмоциональное давление) ---
    frame_len = int(sr * 0.025)  # кадры по 25 мс
    hop_len = int(sr * 0.010)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)
    # Большая вариация → резкие перепады громкости, напор; легитимные: ~0.25, мошенники: 0.35+
    rms_std = float(np.std(rms_norm))
    energy_score = float(np.clip((rms_std - 0.20) / 0.20, 0.0, 1.0))
    scores.append(energy_score)

    # --- 4. Zero-crossing rate (косвенно отражает стресс в голосе) ---
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
    zcr_mean = float(np.mean(zcr))
    # Высокий ZCR → больше высокочастотных компонент; характерен для напряжённой речи
    zcr_score = float(np.clip((zcr_mean - 0.05) / 0.10, 0.0, 1.0))
    scores.append(zcr_score)

    return float(np.mean(scores)) if scores else 0.5
