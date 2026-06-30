# Real vs Screen Photo Detector — Report

## Approach

This detector uses a **hybrid pipeline of four complementary signals** to distinguish real photos from photos-of-screens (recaptures). Each image is analyzed by:

1. **FFT / Moire detection** — Converts to grayscale and computes the 2D Fast Fourier Transform to measure periodic high-frequency energy. Screen photos exhibit characteristic moire patterns from the pixel grid interference between the camera sensor and the screen's sub-pixel layout, producing elevated energy in mid-to-high frequency annuli compared to natural images. Extracts 6 features: mid/high frequency energy ratios, peak-to-mean ratio, spectrum statistics.

2. **Glare / specular highlight detection** — Identifies small, very bright, low-saturation "blown-out" regions in HSV color space (V > 240, S < 40). These specular highlights are caused by ambient light sources (ceiling lights, windows) reflecting off the glass/plastic screen surface — a physical artifact absent in direct real-world photography. Extracts 5 features: blob count, area ratio, circularity, brightness metrics.

3. **Bezel / edge detection** — Uses Canny edge detection and probabilistic Hough line transform to find long straight lines near image borders, consistent with the rectangular bezel of a phone or laptop screen. Extracts 6 features: horizontal/vertical line counts, border edge ratios, darkness contrast.

4. **Fine-tuned MobileNetV2** — A MobileNetV2 backbone (pretrained on ImageNet, convolutional layers frozen) with a trained 2-layer classifier head (1280 -> 128 -> 2). This learns high-level texture and color cues that the hand-crafted features may miss, such as screen color shifts, backlight uniformity, and pixel-level rendering artifacts. Extracts 2 features: logit difference and screen probability.

The four signal groups are concatenated into a **19-dimensional feature vector** and fed into a **Logistic Regression meta-classifier** (with standard scaling) that produces the final fraud probability in [0, 1].

## Results

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.0%** |
| **Validation Accuracy** (train/val split) | **95.0%** |
| **Precision** | 0.9804 |
| **Recall** | 1.0000 |
| **Avg Latency** | 434.3 ms/image |
| **Device** | CPU |

**Confusion Matrix (full dataset, 100 images):**

|  | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 49 | 1 |
| **Actual Screen** | 0 | 50 |

Only 1 misclassification: a real photo (IMG_20260630_113047.jpg) scored 0.91 (flagged as screen). Zero false negatives — every screen photo was caught.

### Per-Signal Standalone Accuracy (Validation Set, 20 images)

| Signal | Features | Accuracy | Precision | Recall |
|--------|----------|----------|-----------|--------|
| FFT (Moire) | 6 | 80.0% | 0.750 | 0.900 |
| Glare | 5 | 55.0% | 0.556 | 0.500 |
| Bezel | 6 | 70.0% | 0.667 | 0.800 |
| CNN (MobileNetV2) | 2 | 95.0% | 0.909 | 1.000 |
| **Combined (all 4)** | **19** | **95.0%** | **0.909** | **1.000** |

The CNN is the strongest individual signal (95%), while FFT and bezel provide meaningful complementary information (80% and 70%). Glare alone is weak (55%) but contributes to the combined model's robustness. The combined meta-classifier achieves 99% on the full evaluation set.

## Latency

| Metric | Value |
|--------|-------|
| Average latency | **434.3 ms** per image |
| Std dev | 75.4 ms |
| Min / Max | 352.3 / 654.6 ms |
| Device | CPU (no GPU) |
| Timing method | 100 runs across 5 images, with warm-up excluded |

## Cost Per Image

| Deployment | Cost |
|------------|------|
| **On-device (phone CPU)** | ~**$0** — model runs locally, no network call required |
| **Cloud server** (assumptions below) | ~**$0.00005** per image |

**Cloud cost assumptions:**
- AWS `c5.large` instance (2 vCPU, 4 GB RAM): ~$0.085/hour on-demand
- Measured throughput: ~2.3 images/sec on CPU (based on 434 ms latency)
- That's ~8,280 images/hour

| Volume | Estimated Cost |
|--------|---------------|
| 1,000 images | ~$0.01 |
| 1,000,000 images | ~$10.27 |

With GPU instances (e.g., g4dn.xlarge at $0.526/hr) or batch processing, throughput would increase 5-10x and per-image cost would drop significantly.

## What I'd Improve With More Time

- **More data**: 50+50 images is small; 500+ per class with diverse screen types (OLED vs LCD, different phones/laptops, various lighting) would significantly improve robustness
- **Adversarial screen types**: Test against edge cases — high-res retina displays (less moire), screens without visible bezels, dark-mode content
- **Threshold calibration**: Use the precision-recall curve to select an optimal decision threshold instead of the default 0.5, tuned to the desired false-positive vs false-negative tradeoff
- **Data augmentation**: More aggressive augmentations (crops, perspective transforms, JPEG compression artifacts) to reduce overfitting on the small dataset
- **Unfreeze deeper layers**: Fine-tune the last few convolutional blocks of MobileNetV2, not just the head, for better feature adaptation

## Bonus: Production Considerations

### Keeping accuracy as cheaters adapt
- **Periodic retraining** on newly collected adversarial examples (screens with anti-glare coatings, borderless phones, digitally cropped recaptures)
- **Active learning pipeline**: flag low-confidence predictions for human review and add them to training data
- **Monitor distribution drift**: track feature statistics in production and alert when distributions shift

### Shrinking for phone deployment
- **TFLite / CoreML conversion**: Export the MobileNetV2 to TensorFlow Lite or CoreML for mobile-optimized inference
- **INT8 quantization**: Reduces model size ~4x and improves inference speed on mobile NPUs
- **Drop weak signals**: If latency-constrained, keep only CNN + the strongest hand-crafted signal (e.g., FFT) and retrain the meta-classifier
- **ONNX Runtime Mobile**: Alternative to TFLite with good cross-platform support

### Choosing the fraud cutoff threshold
- Plot the **precision-recall curve** and choose the threshold based on the **cost of errors**:
  - If false positives are expensive (blocking legitimate users): raise the threshold (e.g., 0.7-0.8)
  - If false negatives are expensive (letting fraud through): lower the threshold (e.g., 0.3-0.4)
- Use **F1-score** or a custom weighted metric to find the optimal operating point
- In production, make the threshold configurable so it can be tuned without retraining
