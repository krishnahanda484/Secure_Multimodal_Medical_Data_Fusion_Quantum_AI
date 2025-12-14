"""
Configuration file for Secure Multimodal Medical Data Fusion Project
Sets up all paths, hyperparameters, and global settings
"""

import os
import torch
from pathlib import Path

# ============================================
# PROJECT PATHS
# ============================================
# Cloud-safe project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Optional: datasets not used in Streamlit demo
SOURCE_DATA_ROOT = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Security
SECURITY_KEY_PATH = os.path.join(PROJECT_ROOT, "security.key")
ENCRYPTED_MODEL_PATH = os.path.join(MODELS_DIR, "secure_model.enc")

# ============================================
# CREATE DIRECTORIES
# ============================================
for directory in [PROCESSED_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# DEVICE CONFIGURATION
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================
# DATASET CONFIGURATIONS
# ============================================
DATASET_CONFIG = {
    'bone_fracture': {
        'path': os.path.join(SOURCE_DATA_ROOT, "Bone Fracture/Bone Fracture/Augmented"),
        'classes': ['Simple', 'Comminuted'],
        'extensions': ['.jpg', '.png']
    },
    'padchest': {
        'images': os.path.join(SOURCE_DATA_ROOT, "PadChest/images/images_normalized"),
        'csv': os.path.join(SOURCE_DATA_ROOT, "PadChest/indiana_reports.csv"),
        'max_samples': 500  # Limit for demo
    },
    'ucsf_brain': {
        'path': os.path.join(SOURCE_DATA_ROOT, "UCSF_BrainMetastases_v1.3/UCSF_BrainMetastases_TRAIN"),
        'max_patients': 100,  # Limit for demo
        'modalities': ['FLAIR', 'seg']
    }
}

# ============================================
# IMAGE PROCESSING
# ============================================
IMG_SIZE = (224, 224)  # Standard size for pretrained models
IMG_CHANNELS = 3

# ============================================
# MODEL HYPERPARAMETERS
# ============================================
# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3

# Feature dimensions
IMG_FEATURE_DIM = 768  # ViT output
TEXT_FEATURE_DIM = 768  # BERT output
FUSION_DIM = 512
CLASSIFIER_HIDDEN = 256

# Organ types
ORGAN_TYPES = ['lung', 'bone', 'brain', 'bone_hbf']
NUM_ORGANS = len(ORGAN_TYPES)

# ============================================
# PRETRAINED MODEL PATHS
# ============================================
# These will download from HuggingFace if not present locally
VIT_MODEL = "google/vit-base-patch16-224"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
# Alternative: "emilyalsentzer/Bio_ClinicalBERT"

# ============================================
# TEXT PROCESSING
# ============================================
MAX_TEXT_LENGTH = 256
TEXT_PADDING = 'max_length'

# ============================================
# QUANTUM SETTINGS
# ============================================
N_QUBITS = 4
QUANTUM_BACKEND = "default.qubit"

# ============================================
# SECURITY SETTINGS
# ============================================
ENCRYPTION_ALGORITHM = "AES-256"  # Using Fernet (AES-256)
USE_QUANTUM_SIGNATURE = True

# ============================================
# DATA SPLIT
# ============================================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================
# APP CONFIGURATION
# ============================================
APP_TITLE = "🩺 Secure Multimodal Medical Diagnosis System"
APP_PORT = 8501
MAX_FILE_SIZE_MB = 50

# Login credentials (for demo - should use proper auth in production)
DEMO_CREDENTIALS = {
    'username': 'admin',
    'password': 'admin123'
}

# ============================================
# LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

print("Configuration loaded successfully!")
print(f"Project Root: {PROJECT_ROOT}")

print(f"Models will be saved to: {MODELS_DIR}")
