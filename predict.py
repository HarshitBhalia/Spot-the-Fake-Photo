"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy-loaded globals so the model is only loaded once even if predict()
# is called multiple times (e.g. batch evaluation or server use).
# ---------------------------------------------------------------------------
_cnn_model = None
_meta_bundle = None


def _load_models():
    """Load the trained MobileNetV2 head and meta-classifier from disk."""
    global _cnn_model, _meta_bundle
    if _cnn_model is not None:
        return  # already loaded

    import pickle
    from features import load_trained_mobilenet

    _cnn_model = load_trained_mobilenet("mobilenet_head.pt")

    with open("meta_classifier.pkl", "rb") as f:
        _meta_bundle = pickle.load(f)


def predict(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")

    # Ensure models are loaded (lazy, one-time cost)
    _load_models()

    # Import here to keep top-level imports minimal (as per starter template)
    import numpy as np
    from features import extract_all_features

    # Extract the same 4 feature groups used during training
    feature_vec = extract_all_features(image_path, _cnn_model)

    # Scale features using the same scaler fitted during training
    scaler = _meta_bundle["scaler"]
    clf = _meta_bundle["classifier"]

    feature_scaled = scaler.transform(feature_vec.reshape(1, -1))

    # Get probability of class 1 (screen / fake)
    prob_screen = clf.predict_proba(feature_scaled)[0, 1]

    return float(round(prob_screen, 4))


if __name__ == "__main__":
    print(predict(sys.argv[1]))
