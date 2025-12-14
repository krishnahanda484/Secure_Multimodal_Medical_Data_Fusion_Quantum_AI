"""
Final Project Configuration
Secure Multimodal Medical Data Fusion with Quantum Security
Cloud + Local Safe
"""

import os
import torch
from pathlib import Path

# ============================================
# ABSOLUTE PROJECT ROOT (FIXED)
# ============================================
PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================
# DIRECTORIES
# ============================================
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"

for d in [MODELS_DIR, LOGS_DIR, REPORTS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# DEVICE
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# MODEL FILES (EXPLICIT & SAFE)
# ============================================
FUSION_MODEL_PATH = MODELS_DIR / "best_fusion.pth"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "best_classifier.pth"

# Validate at import time
if not FUSION_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {FUSION_MODEL_PATH}")

if not CLASSIFIER_MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {CLASSIFIER_MODEL_PATH}")

# ============================================
# PRETRAINED MODELS
# ============================================
VIT_MODEL = "google/vit-base-patch16-224"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

# ============================================
# ORGAN TYPES
# ============================================
ORGAN_TYPES = ["brain", "lung", "bone", "bone_hbf"]

# ============================================
# LOGIN (DEMO ONLY)
# ============================================
DEMO_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

# ============================================
# MISC
# ============================================
MAX_TEXT_LENGTH = 256
IMG_SIZE = 224

print("✅ Configuration loaded")
print("📁 Project Root:", PROJECT_ROOT)
print("📦 Models Dir :", MODELS_DIR)
print("🖥️ Device     :", DEVICE)
