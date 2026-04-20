# Vishing Fraud Detector

Алгоритм автоматического определения мошеннических телефонных разговоров (вишинг).

## Описание

Гибридный мульти-сигнальный пайплайн:

```text
WAV-файл
   │
   ▼
[ASR: Vosk (CPU) / faster-whisper (GPU)]  — транскрипция речи
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
├── best_config.json         # Оптимальные веса и порог (подбирает tune_threshold.py)
├── requirements.txt
├── CMakeLists.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Модели скачиваются автоматически при первом запуске:

- `vosk-model-small-ru-0.22` (~45 МБ, CPU) — по умолчанию
- `paraphrase-multilingual-mpnet-base-v2` (~420 МБ, sentence-transformers)
- `faster-whisper large-v3-turbo` (~1.5 ГБ, только при наличии GPU)

## Использование

### Запуск детектора на папке с файлами

```bash
# запускать из корня проекта phone_frauds/
python -m src.run_test --folder vishing/test --output results.csv --asr vosk
```

Если в корне есть `best_config.json`, веса и порог подгружаются автоматически.

Формат результата (`results.csv`):

```csv
filename;label
out_c_18.wav;0
Nout_b_25.wav;1
```

`label: 0` — мошеннический разговор, `1` — легитимный.

### Флаги

| Флаг          | Описание                              | По умолчанию         |
| ------------- | ------------------------------------- | -------------------- |
| `--folder`    | Папка с WAV-файлами                   | обязательный         |
| `--output`    | Путь к CSV (иначе stdout)             | —                    |
| `--asr`       | ASR-движок: `vosk`, `whisper`, `auto` | `auto`               |
| `--config`    | Путь к `best_config.json`             | `best_config.json`   |
| `--model`     | Whisper модель                        | `large-v3-turbo`     |
| `--threshold` | Порог (переопределяет config)         | из config или `0.50` |
| `--workers`   | Параллельные потоки                   | `1`                  |
| `--verbose`   | Подробный вывод по каждому файлу      | выключен             |

### Подбор оптимальных весов

```bash
python -m src.tune_threshold --samples_dir vishing/samples --asr vosk
```

Результат: `best_config.json` с оптимальными весами и метриками.
Следующий запуск `run_test.py` подхватит его автоматически.

## Результаты

Тест на закрытой выборке `vishing/test` (60 файлов: 30 Fraud + 30 NotFraud).  
ASR: Vosk `vosk-model-small-ru-0.22`, CPU.  
Конфигурация: keyword=0.182, semantic=0.818, threshold=0.40.

| Метрика   | Значение          |
| --------- | ----------------- |
| Accuracy  | **68.3%** (41/60) |
| Precision | 63.4%             |
| Recall    | **86.7%**         |
| F1        | 73.2%             |

```text
               Предсказано: Fraud  Предсказано: Legit
True Fraud           TP = 26            FN = 4
True Legit           FP = 15            TN = 15
```

Высокий Recall (86.7%) — алгоритм пропускает лишь 4 мошеннических звонка из 30.
Основные ошибки (FP=15) — легитимные звонки с банковской тематикой, которые
семантически похожи на мошеннические.

> **Ограничение:** результаты получены с Vosk small-ru (CPU). Модель компактная
> и быстрая (~13 сек/файл), но пропускает часть ключевых фраз. С Whisper на GPU
> ожидается более высокое качество транскрипции и точность.

## Требования

- Python 3.9+
- ОЗУ: ≥8 ГБ
- GPU (опционально, для Whisper): CUDA-совместимая
- Время обработки: ~13 сек/файл (Vosk, CPU)

## Зависимости

```text
vosk                  — ASR на CPU (Vosk small-ru)
faster-whisper        — ASR на GPU (Whisper через CTranslate2)
sentence-transformers — семантические эмбеддинги
librosa               — аудио-признаки
torch                 — бэкенд для моделей
scikit-learn          — метрики и утилиты
```
