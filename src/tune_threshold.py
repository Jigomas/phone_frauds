"""
Tune ensemble weights and classification threshold on the labelled dev set.

Usage:
    python tune_threshold.py --samples_dir vishing/samples

The script:
1. Collects all WAV files from samples/Fraud (label=0) and samples/NotFraud (label=1)
2. Runs the detector on each file (without ensembling — raw signal scores)
3. Grid-searches over (w_keyword, w_semantic, w_prosodic, threshold) to maximise F1
4. Prints the confusion matrix + metrics for the best config
5. Saves the best config to best_config.json

Re-use the saved config in run_test.py via --config flag or load it manually.
"""

import argparse
import json
import os
import sys

import numpy as np

from .detector import FraudDetector


def collect_samples(samples_dir: str) -> list[tuple[str, int]]:
    """Returns list of (wav_path, true_label). Fraud=0, NotFraud=1."""
    items: list[tuple[str, int]] = []
    for label, subdir in [(0, "Fraud"), (1, "NotFraud")]:
        folder = os.path.join(samples_dir, subdir)
        if not os.path.isdir(folder):
            print(f"Warning: {folder} not found, skipping.", file=sys.stderr)
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".wav"):
                items.append((os.path.join(folder, fname), label))
    return items


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 0) & (labels == 1)))
    fn = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 1) & (labels == 1)))
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", default="vishing/samples")
    parser.add_argument("--output", default="best_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--asr", default="auto", help="ASR engine: auto|vosk|whisper")
    args = parser.parse_args()

    samples = collect_samples(args.samples_dir)
    if not samples:
        print("No samples found. Check --samples_dir.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples.", file=sys.stderr)

    detector_kwargs: dict = {"asr": args.asr}
    if args.model:
        detector_kwargs["whisper_model"] = args.model

    # threshold=999 → never fires, we only collect raw signal scores
    detector = FraudDetector(weights={"keyword": 1.0, "semantic": 0.0, "prosodic": 0.0},
                             threshold=999.0, **detector_kwargs)

    kw_scores: list[float] = []
    sem_scores: list[float] = []
    pros_scores: list[float] = []
    true_labels: list[int] = []

    errors = 0
    for i, (path, label) in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {os.path.basename(path)} (true={label})", file=sys.stderr)
        r = detector.predict(path)
        if r.error:
            print(f"    ERROR: {r.error}", file=sys.stderr)
            errors += 1
            continue
        kw_scores.append(r.keyword_score)
        sem_scores.append(r.semantic_score)
        pros_scores.append(r.prosodic_score)
        true_labels.append(label)
        print(f"    kw={r.keyword_score:.3f} sem={r.semantic_score:.3f} pros={r.prosodic_score:.3f}",
              file=sys.stderr)

    if errors:
        print(f"\nWarning: {errors} files failed to process.", file=sys.stderr)

    kw = np.array(kw_scores)
    sem = np.array(sem_scores)
    pros = np.array(pros_scores)
    y = np.array(true_labels)

    # Grid search
    best_f1 = -1.0
    best_cfg: dict = {}

    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    threshold_grid = np.arange(0.1, 0.9, 0.05).tolist()

    for wk in weight_grid:
        for ws in weight_grid:
            for wp in weight_grid:
                total = wk + ws + wp
                if total < 1e-6:
                    continue
                wk_n, ws_n, wp_n = wk / total, ws / total, wp / total
                combined = wk_n * kw + ws_n * sem + wp_n * pros
                for thr in threshold_grid:
                    preds = (combined > thr).astype(int)
                    preds = 1 - preds  # invert: score>thr → fraud(0)
                    m = compute_metrics(preds, y)
                    if m["f1"] > best_f1:
                        best_f1 = m["f1"]
                        best_cfg = dict(
                            weights=dict(keyword=round(wk_n, 3),
                                         semantic=round(ws_n, 3),
                                         prosodic=round(wp_n, 3)),
                            threshold=round(thr, 3),
                            metrics=m,
                        )

    print("\n=== Best configuration ===")
    print(f"  Weights : keyword={best_cfg['weights']['keyword']:.3f}  "
          f"semantic={best_cfg['weights']['semantic']:.3f}  "
          f"prosodic={best_cfg['weights']['prosodic']:.3f}")
    print(f"  Threshold: {best_cfg['threshold']:.3f}")
    m = best_cfg["metrics"]
    print(f"  Accuracy : {m['accuracy']:.3f}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall   : {m['recall']:.3f}")
    print(f"  F1       : {m['f1']:.3f}")
    print(f"  Confusion matrix (fraud=0 / legit=1):")
    print(f"    TP={m['tp']}  FP={m['fp']}")
    print(f"    FN={m['fn']}  TN={m['tn']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)
    print(f"\nConfig saved to {args.output}")


if __name__ == "__main__":
    main()
