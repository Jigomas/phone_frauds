"""
CLI test runner: scans a folder of WAV files and outputs a CSV.

Usage:
    python -m src.run_test --folder /path/to/wavs --output results.csv
    python -m src.run_test --folder /path/to/wavs  # prints to stdout

CSV format (semicolon-separated):
    filename;label
    out_c_18.wav;0
    Nout_b_25.wav;1

label: 0 = fraud, 1 = legit

If best_config.json exists in the working directory, weights and threshold
are loaded from it automatically (produced by tune_threshold.py).
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .detector import FraudDetector

DEFAULT_CONFIG = "best_config.json"


def find_wav_files(folder: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".wav"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def load_config(config_path: str) -> dict | None:
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded config from {config_path} "
              f"(weights={cfg['weights']}, threshold={cfg['threshold']})", file=sys.stderr)
        return cfg
    return None


def build_detector(args, cfg: dict | None) -> FraudDetector:
    kwargs: dict = {}
    if cfg:
        kwargs["weights"] = cfg["weights"]
        kwargs["threshold"] = cfg["threshold"]
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    if args.model:
        kwargs["whisper_model"] = args.model
    kwargs["asr"] = args.asr
    return FraudDetector(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Vishing fraud detector")
    parser.add_argument("--folder",    required=True,       help="Folder with WAV files")
    parser.add_argument("--output",    default=None,        help="Output CSV path (default: stdout)")
    parser.add_argument("--config",    default=DEFAULT_CONFIG,
                        help=f"Path to best_config.json (default: {DEFAULT_CONFIG})")
    parser.add_argument("--asr",       default="auto",      help="ASR engine: auto|vosk|whisper (default: auto — vosk on CPU, whisper on GPU)")
    parser.add_argument("--model",     default=None,        help="Whisper model name (only with --asr whisper)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override fraud score threshold")
    parser.add_argument("--workers",   type=int,   default=1,
                        help="Parallel workers (default 1; >1 helps on CPU, limited benefit on GPU)")
    parser.add_argument("--verbose",   action="store_true", help="Per-file signal breakdown")
    args = parser.parse_args()

    wav_files = find_wav_files(args.folder)
    if not wav_files:
        print(f"No WAV files found in {args.folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(wav_files)} WAV file(s) in {args.folder}", file=sys.stderr)

    cfg = load_config(args.config)
    detector = build_detector(args, cfg)

    # results keyed by wav_path to preserve order
    results: dict = {}
    t_start = time.perf_counter()

    if args.workers <= 1:
        for wav_path in wav_files:
            t0 = time.perf_counter()
            result = detector.predict(wav_path)
            results[wav_path] = (result, time.perf_counter() - t0)
    else:
        # CTranslate2 and sentence-transformers release the GIL — ThreadPoolExecutor works.
        # Each thread shares the same loaded model; faster-whisper is thread-safe.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_path = {
                pool.submit(detector.predict, wav_path): wav_path
                for wav_path in wav_files
            }
            for future in as_completed(future_to_path):
                wav_path = future_to_path[future]
                t0 = time.perf_counter()
                result = future.result()
                results[wav_path] = (result, time.perf_counter() - t0)

    # Output in original sorted order
    rows: list[tuple[str, int]] = []
    total_elapsed = time.perf_counter() - t_start

    for wav_path in wav_files:
        result, elapsed = results[wav_path]
        rows.append((result.filename, result.label))

        if result.error:
            print(f"  ERROR {result.filename}: {result.error}", file=sys.stderr)
        elif args.verbose:
            print(
                f"  {result.filename}: label={result.label} "
                f"score={result.fraud_score:.3f} "
                f"(kw={result.keyword_score:.2f} "
                f"sem={result.semantic_score:.2f} "
                f"pros={result.prosodic_score:.2f}) "
                f"[{elapsed:.1f}s]",
                file=sys.stderr,
            )
        else:
            print(f"  {result.filename}: label={result.label} ({elapsed:.1f}s)", file=sys.stderr)

    n = len(rows)
    print(
        f"\nTotal: {n} files in {total_elapsed:.1f}s ({total_elapsed/n:.1f}s avg)",
        file=sys.stderr,
    )

    header = ["filename", "label"]
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        writer = csv.writer(sys.stdout, delimiter=";")
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
