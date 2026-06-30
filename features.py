"""
features.py — Shared feature extraction for the hybrid real-vs-screen detector.

Contains four signal groups:
  1. FFT / Moiré features  (frequency-domain analysis)
  2. Glare / specular highlight features  (bright blob detection)
  3. Bezel / edge features  (Hough line detection near borders)
  4. MobileNetV2 CNN features  (fine-tuned classifier head)

Each extractor returns a small numpy array of floats.
The `extract_all_features()` function concatenates them into one vector.
"""

import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESIZE_DIM = 512          # Resize longest side for CV features (speed)
MOBILENET_SIZE = 224      # MobileNetV2 input size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature dimensions (for reference / sanity checks)
N_FFT_FEATURES = 6
N_GLARE_FEATURES = 5
N_BEZEL_FEATURES = 6
N_CNN_FEATURES = 2        # logit for real, logit for screen → or just probability


# ---------------------------------------------------------------------------
# 1. FFT / Moiré detection
# ---------------------------------------------------------------------------
def extract_fft_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, compute 2D FFT, and extract features that capture
    periodic high-frequency energy — the hallmark of moiré patterns caused
    by photographing a screen's pixel grid.

    Features returned (6 total):
      0: ratio of mid-frequency energy to total energy
      1: ratio of high-frequency energy to total energy
      2: ratio of mid+high to low-frequency energy
      3: peak-to-mean ratio in the magnitude spectrum
      4: standard deviation of log-magnitude spectrum
      5: kurtosis of log-magnitude spectrum (peakedness → periodic spikes)
    """
    # Convert to grayscale and resize for speed
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    # Compute 2D DFT and shift zero-frequency to center
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) + 1e-10  # avoid log(0)

    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    max_radius = min(crow, ccol)

    # Create radial distance map from center
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)

    # Define frequency bands (fractions of max radius)
    low_mask = dist < (max_radius * 0.1)        # DC + very low freq
    mid_mask = (dist >= max_radius * 0.1) & (dist < max_radius * 0.5)
    high_mask = dist >= max_radius * 0.5

    energy_total = np.sum(magnitude ** 2)
    energy_low = np.sum(magnitude[low_mask] ** 2)
    energy_mid = np.sum(magnitude[mid_mask] ** 2)
    energy_high = np.sum(magnitude[high_mask] ** 2)

    # Feature 0-2: energy ratios
    mid_ratio = energy_mid / (energy_total + 1e-10)
    high_ratio = energy_high / (energy_total + 1e-10)
    mid_high_over_low = (energy_mid + energy_high) / (energy_low + 1e-10)

    # Feature 3: peak-to-mean ratio (periodic signals produce sharp peaks)
    log_mag = np.log(magnitude)
    peak_to_mean = np.max(magnitude) / (np.mean(magnitude) + 1e-10)

    # Feature 4-5: statistics of the log-magnitude spectrum
    std_log_mag = np.std(log_mag)
    # Kurtosis: high kurtosis → sharp peaks from periodic patterns
    mean_log = np.mean(log_mag)
    kurtosis = np.mean((log_mag - mean_log) ** 4) / (std_log_mag ** 4 + 1e-10)

    return np.array([
        mid_ratio, high_ratio, mid_high_over_low,
        peak_to_mean, std_log_mag, kurtosis
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 2. Glare / specular highlight detection
# ---------------------------------------------------------------------------
def extract_glare_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect small, very bright, low-saturation blown-out regions that indicate
    specular reflections on a screen surface (e.g. ceiling lights, windows
    reflecting off a phone/laptop screen).

    Features returned (5 total):
      0: number of bright highlight blobs
      1: total highlight area as fraction of image area
      2: mean circularity of highlight blobs (1.0 = perfect circle)
      3: max brightness in HSV Value channel (normalized to [0,1])
      4: fraction of image pixels that are both very bright AND low-saturation
    """
    # Resize for speed
    h, w = img_bgr.shape[:2]
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # Specular highlights: very bright (V > 240) AND low saturation (S < 40)
    # These are the blown-out white spots from reflections
    bright_mask = (v_ch > 240).astype(np.uint8)
    low_sat_mask = (s_ch < 40).astype(np.uint8)
    highlight_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of highlight regions
    contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    total_area = img_small.shape[0] * img_small.shape[1]

    # Filter tiny noise contours (< 20 pixels)
    contours = [c for c in contours if cv2.contourArea(c) > 20]

    num_blobs = len(contours)
    blob_area_total = sum(cv2.contourArea(c) for c in contours)
    area_ratio = blob_area_total / total_area

    # Mean circularity: 4π·area / perimeter²  (1.0 for a perfect circle)
    circularities = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circ = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circ, 1.0))
    mean_circularity = np.mean(circularities) if circularities else 0.0

    max_brightness = float(np.max(v_ch)) / 255.0
    bright_low_sat_fraction = float(np.sum(highlight_mask > 0)) / total_area

    return np.array([
        num_blobs, area_ratio, mean_circularity,
        max_brightness, bright_low_sat_fraction
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Bezel / edge detection
# ---------------------------------------------------------------------------
def extract_bezel_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Detect strong rectangular edges near the image border — the bezel of a
    phone or laptop screen.  Uses Canny edge detection + Hough line transform.

    Features returned (6 total):
      0: number of long horizontal lines (within 10° of horizontal)
      1: number of long vertical lines (within 10° of vertical)
      2: number of lines in the border region (outer 15% of image)
      3: ratio of border-region edge pixels to total edge pixels
      4: mean darkness of a thin border strip (dark bezel → lower value)
      5: darkness contrast: border strip mean vs. center mean
    """
    h, w = img_bgr.shape[:2]
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection (probabilistic for speed)
    min_line_length = int(min(sh, sw) * 0.3)  # at least 30% of image dimension
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_length, maxLineGap=10)

    n_horizontal = 0
    n_vertical = 0
    n_border_lines = 0
    border_margin = 0.15  # outer 15% of the image

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            # Horizontal: angle < 10°
            if angle < 10:
                n_horizontal += 1
            # Vertical: angle > 80°
            elif angle > 80:
                n_vertical += 1

            # Check if line is near the border
            mid_x = (x1 + x2) / 2 / sw
            mid_y = (y1 + y2) / 2 / sh
            near_border = (mid_x < border_margin or mid_x > 1 - border_margin or
                           mid_y < border_margin or mid_y > 1 - border_margin)
            if near_border:
                n_border_lines += 1

    # Edge pixel ratio in border region vs total
    border_mask = np.zeros_like(edges)
    bm_h = int(sh * border_margin)
    bm_w = int(sw * border_margin)
    border_mask[:bm_h, :] = 1       # top strip
    border_mask[-bm_h:, :] = 1      # bottom strip
    border_mask[:, :bm_w] = 1       # left strip
    border_mask[:, -bm_w:] = 1      # right strip

    total_edge_pixels = np.sum(edges > 0) + 1e-10
    border_edge_pixels = np.sum((edges > 0) & (border_mask > 0))
    border_edge_ratio = border_edge_pixels / total_edge_pixels

    # Darkness of border strip vs center
    border_strip_width = max(int(min(sh, sw) * 0.05), 3)
    border_strip = np.concatenate([
        gray[:border_strip_width, :].ravel(),
        gray[-border_strip_width:, :].ravel(),
        gray[:, :border_strip_width].ravel(),
        gray[:, -border_strip_width:].ravel()
    ])
    center_region = gray[sh // 4: 3 * sh // 4, sw // 4: 3 * sw // 4]

    border_darkness = float(np.mean(border_strip)) / 255.0
    center_brightness = float(np.mean(center_region)) / 255.0
    darkness_contrast = center_brightness - border_darkness  # positive = dark border

    return np.array([
        n_horizontal, n_vertical, n_border_lines,
        border_edge_ratio, border_darkness, darkness_contrast
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. MobileNetV2 CNN features
# ---------------------------------------------------------------------------

# Standard ImageNet normalization for MobileNetV2
MOBILENET_TRANSFORM = T.Compose([
    T.Resize((MOBILENET_SIZE, MOBILENET_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def build_mobilenet(num_classes: int = 2) -> nn.Module:
    """
    Load MobileNetV2 pretrained on ImageNet, freeze the convolutional backbone,
    and replace the classifier head with a small trainable head for our task.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze the entire convolutional feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head (original: Linear(1280 → 1000))
    # We use a small 2-layer head: 1280 → 128 → num_classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )

    return model.to(DEVICE)


def load_trained_mobilenet(model_path: str) -> nn.Module:
    """Load a trained MobileNetV2 with our custom head from disk."""
    model = build_mobilenet(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def extract_cnn_features(img_pil: Image.Image, model: nn.Module) -> np.ndarray:
    """
    Run the image through MobileNetV2 and return the output logits
    plus the softmax probability for the 'screen' class.

    Features returned (2 total):
      0: raw logit difference (screen_logit - real_logit)
      1: softmax probability for 'screen' class
    """
    tensor = MOBILENET_TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)  # shape: (1, 2)

    logits_np = logits.cpu().numpy().flatten()  # [real_logit, screen_logit]
    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # logit difference: positive → model thinks screen
    logit_diff = logits_np[1] - logits_np[0]
    screen_prob = probs[1]

    return np.array([logit_diff, screen_prob], dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------
def extract_all_features(image_path: str, cnn_model: nn.Module) -> np.ndarray:
    """
    Extract all 4 feature groups from a single image and concatenate
    them into one feature vector.

    Returns: np.ndarray of shape (N_FFT + N_GLARE + N_BEZEL + N_CNN,)
    """
    # Load image in both formats needed
    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Fallback: convert from PIL if OpenCV can't read the path
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Extract each signal group
    fft_feats = extract_fft_features(img_bgr)
    glare_feats = extract_glare_features(img_bgr)
    bezel_feats = extract_bezel_features(img_bgr)
    cnn_feats = extract_cnn_features(img_pil, cnn_model)

    # Concatenate into one vector
    return np.concatenate([fft_feats, glare_feats, bezel_feats, cnn_feats])


def get_feature_group_indices():
    """Return dict mapping feature group name → slice of indices in the
    concatenated feature vector. Useful for training per-signal classifiers."""
    idx = 0
    groups = {}
    for name, n in [("fft", N_FFT_FEATURES), ("glare", N_GLARE_FEATURES),
                    ("bezel", N_BEZEL_FEATURES), ("cnn", N_CNN_FEATURES)]:
        groups[name] = slice(idx, idx + n)
        idx += n
    return groups
