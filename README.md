# Vishing Fraud Detector

Алгоритм автоматического определения мошеннических телефонных разговоров (вишинг).

## Описание

Гибридный мульти-сигнальный пайплайн:

```text
WAV-файл
   │
   ▼
[faster-whisper ASR]  — транскрипция речи на русском (GPU/CPU)
   │
   ▼
┌──────────────────────────────────────────────────┐
│  Сигнал 1: Keyword Score                         │
│    60+ триггерных слов/фраз с весами             │
│    (реквизиты карты, OTP-коды, "безопасный счёт" │
│     имперсонация банка/ФСБ/МВД, срочность...)    │
│                                                  │
│  Сигнал 2: Semantic Score                        │
│    sentence-transformers cosine similarity       │
│    к шаблонным фразам мошенников/легитимных звонков│
│                                                  │
│  Сигнал 3: Prosodic Score                        │
│    темп речи (слов/сек), доля тишины,            │
│    вариация RMS-энергии, zero-crossing rate      │
└──────────────────────────────────────────────────┘
   │
   ▼
[Взвешенный ансамбль] → порог → label (0=мошенник, 1=легитимный)
```

## Структура проекта

```text
phone_frauds/
├── src/
│   ├── __init__.py          # Пакет: экспортирует FraudDetector
│   ├── detector.py          # FraudDetector — основной класс
│   ├── keywords.py          # Словарь триггеров с весами
│   ├── semantic.py          # Семантический скорер (sentence-transformers)
│   ├── prosodic.py          # Просодические признаки (librosa)
│   ├── run_test.py          # CLI: папка → CSV
│   └── tune_threshold.py    # Подбор весов/порога на dev-выборке
├── vishing/
│   ├── samples/
│   │   ├── Fraud/           # 20 размеченных мошеннических записей
│   │   └── NotFraud/        # 20 размеченных легитимных записей
│   └── test/
│       ├── Fraud/           # 30 тестовых мошеннических записей
│       └── NotFraud/        # 30 тестовых легитимных записей
├── requirements.txt
├── CMakeLists.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

Модели скачиваются автоматически при первом запуске:

- `faster-whisper large-v3-turbo` (~1.5 ГБ)
- `paraphrase-multilingual-mpnet-base-v2` (~420 МБ)

Сменить модель Whisper: `export WHISPER_MODEL=medium`

## Использование

### Тест по папке с файлами

```bash
# Запускать из корня проекта phone_frauds/
python -m src.run_test --folder vishing/test/Fraud --output results.csv
```

Если в корне проекта есть `best_config.json` (после `tune_threshold`),
веса и порог подгружаются автоматически.

Формат результата (`results.csv`):

```csv
filename;label
out_c_18.wav;0
out_c_19.wav;0
```

`label: 0` — мошеннический разговор, `1` — легитимный.

### Флаги

| Флаг          | Описание                              | По умолчанию         |
| ------------- | ------------------------------------- | -------------------- |
| `--folder`    | Папка с WAV-файлами                   | обязательный         |
| `--output`    | Путь к CSV (иначе stdout)             | —                    |
| `--config`    | Путь к `best_config.json`             | `best_config.json`   |
| `--model`     | Whisper модель                        | `large-v3-turbo`     |
| `--threshold` | Порог (переопределяет config)         | из config или `0.50` |
| `--workers`   | Параллельные потоки (CPU: 4+, GPU: 1) | `1`                  |
| `--verbose`   | Подробный вывод по каждому файлу      | выключен             |

### Подбор оптимальных весов (опционально)

```bash
python -m src.tune_threshold --samples_dir vishing/samples --output best_config.json
```

Результат: файл `best_config.json` с оптимальными весами и метриками (accuracy, F1, confusion matrix).
Следующий запуск `run_test.py` подхватит его автоматически.

## Требования

- Python 3.9+
- CUDA-совместимая GPU (опционально, работает и на CPU)
- ОЗУ: ≥8 ГБ
- Время обработки: ~10–30 с / файл на RTX 4090

## Зависимости

```text
faster-whisper        — ASR (Whisper через CTranslate2)
sentence-transformers — семантические эмбеддинги
librosa               — аудио-признаки
torch                 — бэкенд для моделей
scikit-learn          — метрики и утилиты
```
