"""
evaluate.py — Evaluate the trained hybrid detector on the full dataset.

Reports:
  - Overall accuracy, confusion matrix, precision, recall
  - Average latency per image (ms), measured properly with warm-up
  - Device used (CPU or CUDA)
"""

import os
import sys
import time
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix)

# Import predict from predict.py
from predict import predict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = "Datasett"
REAL_DIR = os.path.join(DATA_DIR, "Real")
SCREEN_DIR = os.path.join(DATA_DIR, "Fake")
N_TIMING_RUNS = 20    # number of timed runs per image (after warm-up)


def collect_images():
    """Gather all image paths and ground-truth labels."""
    paths, labels = [], []

    for fname in sorted(os.listdir(REAL_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('.'):
            paths.append(os.path.join(REAL_DIR, fname))
            labels.append(0)  # real

    for fname in sorted(os.listdir(SCREEN_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('.'):
            paths.append(os.path.join(SCREEN_DIR, fname))
            labels.append(1)  # screen / fake

    return paths, labels


def main():
    print("=" * 60)
    print("EVALUATION -- Hybrid Real-vs-Screen Detector")
    print("=" * 60)

    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n[Device] Running on: {device_name}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    paths, labels = collect_images()
    print(f"[Data]   {len(paths)} images ({labels.count(0)} real, "
          f"{labels.count(1)} screen)")

    # ----- Accuracy evaluation -----
    print("\n[Eval]   Running predictions...")
    predictions = []
    pred_labels = []

    for i, path in enumerate(paths):
        score = predict(path)
        predictions.append(score)
        pred_labels.append(1 if score >= 0.5 else 0)
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            print(f"  Processed {i + 1}/{len(paths)}", end="\r")
    print()

    y_true = np.array(labels)
    y_pred = np.array(pred_labels)
    y_scores = np.array(predictions)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'-' * 40}")
    print(f"  Overall Accuracy:  {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  Precision:         {prec:.4f}")
    print(f"  Recall:            {rec:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    Predicted ->   Real   Screen")
    print(f"    Actual Real   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"    Actual Screen {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    # Show misclassified images for debugging
    misclassified = [(paths[i], labels[i], predictions[i])
                     for i in range(len(paths)) if pred_labels[i] != labels[i]]
    if misclassified:
        print(f"\n  Misclassified ({len(misclassified)} images):")
        for path, true_label, score in misclassified:
            label_str = "real" if true_label == 0 else "screen"
            print(f"    {os.path.basename(path)} -- true: {label_str}, "
                  f"score: {score:.4f}")

    # ----- Latency measurement -----
    print(f"\n{'-' * 40}")
    print(f"[Latency] Measuring with {N_TIMING_RUNS} timed runs per image "
          f"(+ 1 warm-up)...")

    # Pick a subset of images for timing (use up to 5 images, repeat N times)
    timing_images = paths[:min(5, len(paths))]

    # Warm-up run (excluded from timing)
    for img_path in timing_images:
        _ = predict(img_path)

    # Timed runs
    all_times = []
    for run in range(N_TIMING_RUNS):
        for img_path in timing_images:
            t_start = time.perf_counter()
            _ = predict(img_path)
            t_end = time.perf_counter()
            all_times.append(t_end - t_start)

    avg_latency_ms = np.mean(all_times) * 1000
    std_latency_ms = np.std(all_times) * 1000
    min_latency_ms = np.min(all_times) * 1000
    max_latency_ms = np.max(all_times) * 1000

    print(f"  Average latency:  {avg_latency_ms:.1f} ms per image")
    print(f"  Std dev:          {std_latency_ms:.1f} ms")
    print(f"  Min / Max:        {min_latency_ms:.1f} / {max_latency_ms:.1f} ms")
    print(f"  Device:           {device_name}")
    print(f"  Timing samples:   {len(all_times)} runs across "
          f"{len(timing_images)} images")

    # ----- Summary -----
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc * 100:.1f}%")
    print(f"  Latency:   {avg_latency_ms:.1f} ms/image on {device_name}")

    if acc >= 0.95:
        print("  [PASS] Meets the >95% accuracy target")
    else:
        print(f"  [FAIL] Below the >95% target ({acc * 100:.1f}%)")


if __name__ == "__main__":
    main()
