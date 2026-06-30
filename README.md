# Spot the Fake Photo: Hybrid Anti-Spoofing Detector

![Accuracy](https://img.shields.io/badge/Accuracy-99.0%25-success)
![Latency](https://img.shields.io/badge/Latency-434ms-blue)
![Platform](https://img.shields.io/badge/Platform-CPU_Optimized-lightgrey)
![Framework](https://img.shields.io/badge/Framework-PyTorch_%7C_Flask-orange)

## 📌 Project Overview
This project is an advanced, lightweight computer vision pipeline designed to solve a critical anti-spoofing problem: **detecting whether a given image is a genuine, real-world photograph or a "recapture"** (a photo taken of a phone/laptop screen displaying an image). 

Such systems are vital for KYC (Know Your Customer) identity verification, fraud prevention, and maintaining the integrity of image-based data submissions.

The detector achieves a **99.0% accuracy** on a custom dataset using a highly optimized, hybrid 4-signal approach that fuses classical image processing techniques with a fine-tuned deep learning model, explicitly designed to run efficiently on CPU devices (like mobile phones or lightweight cloud servers).

---

## 🔬 The 4-Signal Hybrid Architecture

Instead of relying solely on a black-box deep learning model, this project extracts specific physical and digital anomalies caused by photographing a screen. The system extracts a **19-dimensional feature vector** comprised of four distinct signals:

### 1. FFT / Moiré Detection (Frequency Domain)
When a camera sensor captures an image of a digital screen, the interference between the camera's pixel grid and the screen's sub-pixel layout creates distinct high-frequency aliasing known as moiré patterns. 
- **How it works:** The image is converted to grayscale, and a 2D Fast Fourier Transform (FFT) is applied.
- **Features extracted:** Ratios of energy in mid/high frequency annuli vs total energy, peak-to-mean ratio in the frequency spectrum, and log-magnitude kurtosis.

### 2. Specular Glare Detection (Color Space)
Screens are typically covered in glossy glass or plastic, which reflects ambient light (windows, ceiling lights) as harsh, blown-out white spots.
- **How it works:** The image is converted to HSV color space to identify extremely bright (`V > 240`) and heavily desaturated (`S < 40`) localized blobs.
- **Features extracted:** Highlight blob count, total relative area, average circularity, and maximum brightness.

### 3. Bezel & Edge Detection (Spatial Geometry)
Photos of screens often inadvertently capture the straight, dark rectangular borders (bezels) of the device.
- **How it works:** Canny edge detection combined with a Probabilistic Hough Line Transform is used to find long, perfectly straight lines running parallel to the image borders.
- **Features extracted:** Count of vertical/horizontal lines near the edge, edge pixel density in the border region, and center-to-border darkness contrast.

### 4. Fine-Tuned MobileNetV2 (Deep Learning)
To capture subtle, high-level cues that handcrafted features might miss (e.g., backlight bleeding, color shifting, sub-pixel rendering artifacts), a lightweight CNN is used.
- **How it works:** A PyTorch `MobileNetV2` backbone (pretrained on ImageNet, with frozen convolutional layers) is equipped with a custom 2-layer classifier head. The head is fine-tuned specifically on our dataset.
- **Features extracted:** The final classification logits and softmax probabilities.

**Meta-Classifier:** 
The 19 features extracted from these 4 signals are standardized and fed into a **Logistic Regression** meta-classifier, which makes the final prediction (0.0 = Real, 1.0 = Screen).

---

## 📊 Dataset Creation

A custom dataset was meticulously collected to train and evaluate this model:
* **Total Images:** 100 high-resolution images (`.jpg`).
* **Real Photos (50 images):** Genuine photographs of various subjects (people, documents, objects, outdoors) taken directly with a smartphone camera.
* **Screen Recaptures (50 images):** Photos taken of laptops and secondary smartphones displaying images. These include various challenging scenarios: different brightness levels, angles, and screen technologies (OLED vs LCD).

The dataset is located in the `Datasett/` directory, split into `Real/` and `Fake/` subfolders.

---

## 🏆 Results & Performance

The model was rigorously evaluated on the dataset using a stratified train/validation split, yielding the following results:

| Metric | Value |
|--------|-------|
| **Overall Evaluation Accuracy** | **99.0%** (99/100 correct) |
| **Validation Accuracy (80/20 split)**| **95.0%** |
| **Precision** | **0.9804** |
| **Recall** (Sensitivity) | **1.000** (Zero false negatives) |
| **Average CPU Latency** | **~434 ms** per image |

**Per-Signal Contribution (Standalone Validation Accuracy):**
* MobileNetV2 CNN: `95.0%`
* FFT (Moiré): `80.0%`
* Bezel (Edge): `70.0%`
* Glare (Specular): `55.0%`

*Note: While Glare and Bezel features are weaker on their own, their inclusion increases the overall robustness of the meta-classifier against edge cases.*

---

## 💻 Web Demo & Deployment

This project includes a beautiful, responsive, glassmorphic **Flask Web Interface** allowing users to upload an image and get an instant fraud probability score.

### Running the App Locally
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the Web Server:**
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your web browser.

### Cloud Deployment (Render / Heroku)
The repository is 100% configured for a 1-click PaaS deployment (e.g., Render).
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn app:app`

*(Note: The project uses `opencv-python-headless` in `requirements.txt` to ensure smooth server deployments without X11 GUI dependencies).*

---

## 🛠️ Repository Structure

* `features.py`: The core signal extraction logic (FFT, Glare, Bezel, CNN).
* `train.py`: Script to train the MobileNetV2 head, extract features, and train the meta-classifier.
* `predict.py`: A lightweight, one-line predictor script for CLI usage.
* `evaluate.py`: Tests the models against the entire dataset to compute accuracy, precision, recall, and exact execution latency.
* `app.py`: The Flask backend for the web demo.
* `templates/index.html`: The frontend UI for the web demo.
* `mobilenet_head.pt` & `meta_classifier.pkl`: The saved, trained model weights.
* `report.md`: The original half-page assignment write-up detailing costs and future improvements.
