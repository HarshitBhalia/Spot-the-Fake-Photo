"""
train.py — Train the hybrid real-vs-screen photo detector.

Steps:
  1. Load images from Datasett/Real/ and Datasett/Fake/
  2. Fine-tune MobileNetV2 head on the dataset
  3. Extract all 4 feature groups for every image using the trained CNN
  4. Train a Logistic Regression meta-classifier on the combined features
  5. Report per-signal and combined accuracy on a stratified validation split
  6. Save: mobilenet_head.pt  and  meta_classifier.pkl
"""

import os
import sys
import random
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, classification_report)

from features import (
    build_mobilenet, extract_all_features, extract_fft_features,
    extract_glare_features, extract_bezel_features, extract_cnn_features,
    MOBILENET_TRANSFORM, DEVICE, get_feature_group_indices,
    load_trained_mobilenet
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
DATA_DIR = os.path.join("Datasett")
REAL_DIR = os.path.join(DATA_DIR, "Real")
SCREEN_DIR = os.path.join(DATA_DIR, "Fake")
VAL_SPLIT = 0.2           # 80/20 stratified split
CNN_EPOCHS = 15            # epochs for fine-tuning the MobileNetV2 head
CNN_LR = 1e-3              # learning rate for the classifier head
CNN_BATCH = 8
MODEL_SAVE_PATH = "mobilenet_head.pt"
META_SAVE_PATH = "meta_classifier.pkl"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset for MobileNetV2 fine-tuning
# ---------------------------------------------------------------------------
class ScreenDataset(Dataset):
    """Simple dataset: loads image paths + labels, applies MobileNet transforms."""

    def __init__(self, paths, labels, transform=None, augment=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        # Augmentation transforms for training
        self.aug_transform = torch.nn.Sequential() if not augment else None
        if augment:
            import torchvision.transforms as T
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.augment:
            tensor = self.aug_transform(img)
        elif self.transform:
            tensor = self.transform(img)
        else:
            tensor = MOBILENET_TRANSFORM(img)
        return tensor, label


# ---------------------------------------------------------------------------
# Collect image paths and labels
# ---------------------------------------------------------------------------
def collect_data():
    """Scan data directories and return lists of (path, label) pairs.
    Label 0 = real, Label 1 = screen/fake."""
    paths, labels = [], []

    for fname in sorted(os.listdir(REAL_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('.'):
            paths.append(os.path.join(REAL_DIR, fname))
            labels.append(0)

    for fname in sorted(os.listdir(SCREEN_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')) and not fname.startswith('.'):
            paths.append(os.path.join(SCREEN_DIR, fname))
            labels.append(1)

    print(f"[Data] Found {labels.count(0)} real + {labels.count(1)} screen = "
          f"{len(labels)} total images")
    return paths, labels


# ---------------------------------------------------------------------------
# Phase 1: Fine-tune MobileNetV2 classifier head
# ---------------------------------------------------------------------------
def train_mobilenet(train_paths, train_labels, val_paths, val_labels):
    """Fine-tune only the classifier head of MobileNetV2."""
    print("\n" + "=" * 60)
    print("PHASE 1: Fine-tuning MobileNetV2 classifier head")
    print("=" * 60)

    model = build_mobilenet(num_classes=2)
    print(f"[CNN] Device: {DEVICE}")
    print(f"[CNN] Training on {len(train_paths)} images, "
          f"validating on {len(val_paths)} images")

    # Create datasets with augmentation for training
    train_ds = ScreenDataset(train_paths, train_labels,
                             augment=True)
    val_ds = ScreenDataset(val_paths, val_labels,
                           transform=MOBILENET_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False,
                            num_workers=0)

    # Only optimize the classifier head parameters (backbone is frozen)
    optimizer = optim.Adam(model.classifier.parameters(), lr=CNN_LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(CNN_EPOCHS):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_imgs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_imgs.size(0)

        # --- Validation ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                outputs = model(batch_imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_imgs.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / train_total

        print(f"  Epoch {epoch + 1:2d}/{CNN_EPOCHS} - "
              f"loss: {avg_loss:.4f}  train_acc: {train_acc:.3f}  "
              f"val_acc: {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    print(f"\n[CNN] Best validation accuracy: {best_val_acc:.3f}")

    # Save the full model state (backbone + head)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[CNN] Saved model to {MODEL_SAVE_PATH}")

    return model


# ---------------------------------------------------------------------------
# Phase 2: Extract all features and train meta-classifier
# ---------------------------------------------------------------------------
def extract_features_for_dataset(paths, labels, cnn_model):
    """Extract the full feature vector for every image in the dataset."""
    import cv2
    features_list = []
    for i, path in enumerate(paths):
        feats = extract_all_features(path, cnn_model)
        features_list.append(feats)
        if (i + 1) % 10 == 0 or (i + 1) == len(paths):
            print(f"  Extracted features: {i + 1}/{len(paths)}", end="\r")
    print()
    return np.array(features_list)


def train_meta_classifier(X_train, y_train, X_val, y_val):
    """Train a Logistic Regression meta-classifier on the combined features."""
    print("\n" + "=" * 60)
    print("PHASE 2: Training meta-classifier (Logistic Regression)")
    print("=" * 60)

    # Standardize features (fit on train, transform both)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Logistic Regression with regularization
    clf = LogisticRegression(
        C=1.0,              # regularization strength (inverse)
        max_iter=1000,
        solver='lbfgs',
        random_state=SEED,
    )
    clf.fit(X_train_scaled, y_train)

    # Predict on validation set
    y_pred = clf.predict(X_val_scaled)
    y_prob = clf.predict_proba(X_val_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    print(f"\n[Meta] Validation Accuracy:  {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"[Meta] Precision:            {prec:.4f}")
    print(f"[Meta] Recall:               {rec:.4f}")
    print(f"\n[Meta] Confusion Matrix:")
    print(f"  Predicted ->    Real   Screen")
    print(f"  Actual Real   {cm[0, 0]:5d}   {cm[0, 1]:5d}")
    print(f"  Actual Screen {cm[1, 0]:5d}   {cm[1, 1]:5d}")

    # Save classifier + scaler together
    meta_bundle = {"classifier": clf, "scaler": scaler}
    with open(META_SAVE_PATH, "wb") as f:
        pickle.dump(meta_bundle, f)
    print(f"\n[Meta] Saved meta-classifier to {META_SAVE_PATH}")

    return clf, scaler, acc


# ---------------------------------------------------------------------------
# Per-signal standalone accuracy
# ---------------------------------------------------------------------------
def report_per_signal_accuracy(X_train, y_train, X_val, y_val):
    """Train and evaluate a separate Logistic Regression for each signal group
    to show each component's individual contribution."""
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print("PER-SIGNAL STANDALONE ACCURACY (on validation set)")
    print("=" * 60)

    groups = get_feature_group_indices()
    for name, idx in groups.items():
        X_tr = X_train[:, idx]
        X_va = X_val[:, idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_tr_s, y_train)
        y_pred = clf.predict(X_va_s)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        n_features = X_tr.shape[1]
        print(f"  {name.upper():6s} ({n_features} features): "
              f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)

    # Step 1: Collect data
    paths, labels = collect_data()

    # Step 2: Stratified train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=VAL_SPLIT, stratify=labels, random_state=SEED
    )
    print(f"[Split] Train: {len(train_paths)}  Val: {len(val_paths)}")

    # Step 3: Fine-tune MobileNetV2 classifier head
    cnn_model = train_mobilenet(train_paths, train_labels, val_paths, val_labels)

    # Step 4: Extract features for all images (using the trained CNN)
    print("\n[Features] Extracting features for training set...")
    X_train = extract_features_for_dataset(train_paths, train_labels, cnn_model)
    print("[Features] Extracting features for validation set...")
    X_val = extract_features_for_dataset(val_paths, val_labels, cnn_model)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    print(f"[Features] Feature vector dimension: {X_train.shape[1]}")

    # Step 5: Report per-signal standalone accuracy
    report_per_signal_accuracy(X_train, y_train, X_val, y_val)

    # Step 6: Train meta-classifier on combined features
    clf, scaler, val_acc = train_meta_classifier(X_train, y_train, X_val, y_val)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  MobileNetV2 head saved to: {MODEL_SAVE_PATH}")
    print(f"  Meta-classifier saved to:  {META_SAVE_PATH}")
    print(f"  Final validation accuracy:  {val_acc * 100:.1f}%")

    # Improvement suggestions if accuracy is below target
    if val_acc < 0.95:
        print("\n  WARNING: Validation accuracy is below 95% target.")
        print("  Suggestions to improve:")
        print("    - Collect more training data (especially edge cases)")
        print("    - Add more aggressive data augmentation")
        print("    - Tune the decision threshold via precision-recall curve")
        print("    - Unfreeze last few conv layers of MobileNetV2 for fine-tuning")
        print("    - Try an ensemble of classifiers instead of single LR")


if __name__ == "__main__":
    main()
