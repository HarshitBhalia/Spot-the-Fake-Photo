import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

RESIZE_DIM = 512

def extract_fft_features(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)

    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) + 1e-10

    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    max_radius = min(crow, ccol)

    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)

    low_mask = dist < (max_radius * 0.1)
    mid_mask = (dist >= max_radius * 0.1) & (dist < max_radius * 0.5)
    high_mask = dist >= max_radius * 0.5

    energy_total = np.sum(magnitude ** 2)
    energy_low = np.sum(magnitude[low_mask] ** 2)
    energy_mid = np.sum(magnitude[mid_mask] ** 2)
    energy_high = np.sum(magnitude[high_mask] ** 2)

    mid_ratio = energy_mid / (energy_total + 1e-10)
    high_ratio = energy_high / (energy_total + 1e-10)
    mid_high_over_low = (energy_mid + energy_high) / (energy_low + 1e-10)

    log_mag = np.log(magnitude)
    peak_to_mean = np.max(magnitude) / (np.mean(magnitude) + 1e-10)

    std_log_mag = np.std(log_mag)
    mean_log = np.mean(log_mag)
    kurtosis = np.mean((log_mag - mean_log) ** 4) / (std_log_mag ** 4 + 1e-10)

    return np.array([
        mid_ratio, high_ratio, mid_high_over_low,
        peak_to_mean, std_log_mag, kurtosis
    ], dtype=np.float32)


def extract_glare_features(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    bright_mask = (v_ch > 240).astype(np.uint8)
    low_sat_mask = (s_ch < 40).astype(np.uint8)
    highlight_mask = cv2.bitwise_and(bright_mask, low_sat_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    highlight_mask = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    total_area = img_small.shape[0] * img_small.shape[1]

    contours = [c for c in contours if cv2.contourArea(c) > 20]

    num_blobs = len(contours)
    blob_area_total = sum(cv2.contourArea(c) for c in contours)
    area_ratio = blob_area_total / total_area

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


def extract_bezel_features(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = RESIZE_DIM / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr.copy()

    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_length = int(min(sh, sw) * 0.3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=min_line_length, maxLineGap=10)

    n_horizontal = 0
    n_vertical = 0
    n_border_lines = 0
    border_margin = 0.15

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            if angle < 10:
                n_horizontal += 1
            elif angle > 80:
                n_vertical += 1

            mid_x = (x1 + x2) / 2 / sw
            mid_y = (y1 + y2) / 2 / sh
            near_border = (mid_x < border_margin or mid_x > 1 - border_margin or
                           mid_y < border_margin or mid_y > 1 - border_margin)
            if near_border:
                n_border_lines += 1

    border_mask = np.zeros_like(edges)
    bm_h = int(sh * border_margin)
    bm_w = int(sw * border_margin)
    border_mask[:bm_h, :] = 1
    border_mask[-bm_h:, :] = 1
    border_mask[:, :bm_w] = 1
    border_mask[:, -bm_w:] = 1

    total_edge_pixels = np.sum(edges > 0) + 1e-10
    border_edge_pixels = np.sum((edges > 0) & (border_mask > 0))
    border_edge_ratio = border_edge_pixels / total_edge_pixels

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
    darkness_contrast = center_brightness - border_darkness

    return np.array([
        n_horizontal, n_vertical, n_border_lines,
        border_edge_ratio, border_darkness, darkness_contrast
    ], dtype=np.float32)


class ONNXMobileNet:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def __call__(self, img_path):
        img = Image.open(img_path).convert('RGB')
        
        img = img.resize((224, 224), Image.BILINEAR)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        
        img_np = np.transpose(img_np, (2, 0, 1))
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_np = (img_np - mean) / std
        
        x = np.expand_dims(img_np, axis=0)
        
        outs = self.session.run(None, {self.input_name: x})
        logits = outs[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        logit_diff = logits[1] - logits[0]
        screen_prob = probs[1]
        return np.array([logit_diff, screen_prob], dtype=np.float32)

def extract_all_features_deploy(img_path: str, cnn_model) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    fft_feat = extract_fft_features(img_bgr)
    glare_feat = extract_glare_features(img_bgr)
    bezel_feat = extract_bezel_features(img_bgr)
    cnn_feat = cnn_model(img_path)
    
    return np.concatenate([fft_feat, glare_feat, bezel_feat, cnn_feat])
