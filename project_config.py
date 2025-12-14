"""
Configuration file for Secure Multimodal Medical Data Fusion Project
Cloud-safe and Streamlit-compatible
"""

import os
import torch
from pathlib import Path

# ============================================
# PROJECT ROOT (CLOUD SAFE)
# ============================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================
# DATA & DIRECTORY PATHS
# ============================================
SOURCE_DATA_ROOT = PROJECT_ROOT / "data"          # Optional (not required for demo)
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ============================================
# SECURITY PATHS
# ============================================
SECURITY_KEY_PATH = PROJECT_ROOT / "security.key"
ENCRYPTED_MODEL_PATH = MODELS_DIR / "secure_model.enc"

# ============================================
# CREATE REQUIRED DIRECTORIES
# ============================================
for directory in [
    PROCESSED_DIR,
    MODELS_DIR,
    LOGS_DIR,
    REPORTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# DEVICE CONFIGURATION
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================
# DATASET CONFIGURATION (OPTIONAL / TRAINING)
# ============================================
DATASET_CONFIG = {
    "bone_fracture": {
        "path": SOURCE_DATA_ROOT / "bone_fracture",
        "classes": ["Simple", "Comminuted"],
        "extensions": [".jpg", ".png"]
    },
    "padchest": {
        "images": SOURCE_DATA_ROOT / "padchest/images",
        "csv": SOURCE_DATA_ROOT / "padchest/reports.csv",
        "max_samples": 500
    },
    "ucsf_brain": {
        "path": SOURCE_DATA_ROOT / "ucsf_brain",
        "max_patients": 100,
        "modalities": ["FLAIR", "seg"]
    }
}

# ============================================
# IMAGE PROCESSING
# ============================================
IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

# ============================================
# MODEL HYPERPARAMETERS
# ============================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3

IMG_FEATURE_DIM = 768
TEXT_FEATURE_DIM = 768
FUSION_DIM = 512
CLASSIFIER_HIDDEN = 256

# ============================================
# ORGAN CONFIGURATION
# ============================================
ORGAN_TYPES = ["lung", "bone", "brain", "bone_hbf"]
NUM_ORGANS = len(ORGAN_TYPES)

# ============================================
# PRETRAINED MODELS (HUGGINGFACE)
# ============================================
VIT_MODEL = "google/vit-base-patch16-224"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

# ============================================
# TEXT PROCESSING
# ============================================
MAX_TEXT_LENGTH = 256
TEXT_PADDING = "max_length"

# ============================================
# QUANTUM SETTINGS
# ============================================
N_QUBITS = 4
QUANTUM_BACKEND = "default.qubit"

# ============================================
# SECURITY SETTINGS
# ============================================
ENCRYPTION_ALGORITHM = "AES-256"
USE_QUANTUM_SIGNATURE = True

# ============================================
# DATA SPLITS
# ============================================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# ============================================
# APP SETTINGS
# ============================================
APP_TITLE = "🩺 Secure Multimodal Medical Diagnosis System"
APP_PORT = 8501
MAX_FILE_SIZE_MB = 50

# ============================================
# DEMO LOGIN (STREAMLIT)
# ============================================
DEMO_CREDENTIALS = {
    "username": "admin",
    "password": "admin123"
}

# ============================================
# LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================
# FINAL STATUS LOGS
# ============================================
print("Configuration loaded successfully!")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Models directory: {MODELS_DIR}")
