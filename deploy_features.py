import cv2
import numpy as np
from PIL import Image
import math
import onnxruntime as ort
import pickle
import os

def extract_fft_features(img_gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    max_r = min(h, w) // 2
    mask_low = r < (max_r * 0.3)
    mask_mid = (r >= (max_r * 0.3)) & (r < (max_r * 0.7))
    mask_high = r >= (max_r * 0.7)
    
    total_energy = np.sum(magnitude_spectrum) + 1e-8
    energy_low = np.sum(magnitude_spectrum[mask_low]) / total_energy
    energy_mid = np.sum(magnitude_spectrum[mask_mid]) / total_energy
    energy_high = np.sum(magnitude_spectrum[mask_high]) / total_energy
    
    peak_to_mean = np.max(magnitude_spectrum) / (np.mean(magnitude_spectrum) + 1e-8)
    std_dev = np.std(magnitude_spectrum)
    kurtosis = np.mean(((magnitude_spectrum - np.mean(magnitude_spectrum)) / (std_dev + 1e-8))**4)
    
    return np.array([energy_low, energy_mid, energy_high, peak_to_mean, std_dev, kurtosis])

def extract_glare_features(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = cv2.inRange(hsv, (0, 0, 240), (180, 40, 255))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blob_count = len(contours)
    total_area = sum(cv2.contourArea(c) for c in contours)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    area_ratio = total_area / img_area
    
    circularity_sum = 0
    valid_blobs = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10:
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                circularity_sum += circularity
                valid_blobs += 1
                
    avg_circularity = circularity_sum / valid_blobs if valid_blobs > 0 else 0
    max_v = np.max(v)
    mean_v_glare = np.mean(v[mask > 0]) if np.any(mask > 0) else 0
    
    return np.array([blob_count, area_ratio, avg_circularity, max_v, mean_v_glare])

def extract_bezel_features(img_gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    h, w = img_gray.shape
    margin = int(min(h, w) * 0.15)
    
    h_lines = 0
    v_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 10 or abs(angle) > 170:
                if y1 < margin or y1 > h - margin:
                    h_lines += 1
            elif abs(angle) > 80 and abs(angle) < 100:
                if x1 < margin or x1 > w - margin:
                    v_lines += 1
                    
    edge_density_top = np.mean(edges[0:margin, :]) / 255.0
    edge_density_bottom = np.mean(edges[h-margin:h, :]) / 255.0
    edge_density_left = np.mean(edges[:, 0:margin]) / 255.0
    edge_density_right = np.mean(edges[:, w-margin:w]) / 255.0
    border_edge_ratio = (edge_density_top + edge_density_bottom + edge_density_left + edge_density_right) / 4.0
    
    center_mean = np.mean(img_gray[margin:h-margin, margin:w-margin])
    border_mean = (np.mean(img_gray[0:margin, :]) + np.mean(img_gray[h-margin:h, :]) + 
                   np.mean(img_gray[:, 0:margin]) + np.mean(img_gray[:, w-margin:w])) / 4.0
    contrast = abs(center_mean - border_mean)
    
    return np.array([h_lines, v_lines, border_edge_ratio, contrast, edge_density_top, edge_density_bottom])

class ONNXMobileNet:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
    def __call__(self, img_path):
        # Preprocess using pure Numpy/Pillow instead of torchvision
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # HWC to CHW
        img_np = np.transpose(img_np, (2, 0, 1))
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_np = (img_np - mean) / std
        
        # Add batch dimension
        x = np.expand_dims(img_np, axis=0)
        
        # Infer
        outs = self.session.run(None, {self.input_name: x})
        logits = outs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        logit_diff = logits[1] - logits[0]
        screen_prob = probs[1]
        return np.array([logit_diff, screen_prob])

def extract_all_features_deploy(img_path: str, cnn_model) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")
        
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    fft_feat = extract_fft_features(img_gray)
    glare_feat = extract_glare_features(img_bgr)
    bezel_feat = extract_bezel_features(img_gray)
    cnn_feat = cnn_model(img_path)
    
    return np.concatenate([fft_feat, glare_feat, bezel_feat, cnn_feat])
