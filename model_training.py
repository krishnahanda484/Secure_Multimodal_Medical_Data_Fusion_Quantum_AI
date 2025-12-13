"""
Model Training Module
Implements attention-based fusion and trains multimodal classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from project_config import (
    PROJECT_ROOT,
    MODELS_DIR,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    DROPOUT_RATE,
    IMG_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    FUSION_DIM,
    CLASSIFIER_HIDDEN,
    NUM_ORGANS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# DATASET CLASS
# ============================================
class MultimodalDataset(Dataset):
    """Dataset for multimodal medical data"""
    
    def __init__(self, features_path):
        data = np.load(features_path)
        self.img_features = torch.FloatTensor(data['img_features'])
        self.txt_features = torch.FloatTensor(data['txt_features'])
        self.labels = torch.FloatTensor(data['labels'])
        self.organ_indices = torch.LongTensor(data['organ_indices'])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'img_feat': self.img_features[idx],
            'txt_feat': self.txt_features[idx],
            'label': self.labels[idx],
            'organ_idx': self.organ_indices[idx]
        }


# ============================================
# ATTENTION-BASED FUSION MODEL
# ============================================
class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism
    Learns to weight image and text features dynamically
    """
    
    def __init__(self, img_dim=IMG_FEATURE_DIM, txt_dim=TEXT_FEATURE_DIM, fusion_dim=FUSION_DIM):
        super(AttentionFusion, self).__init__()
        
        # Project features to common dimension
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.txt_proj = nn.Linear(txt_dim, fusion_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, 2),  # 2 attention weights (img, txt)
            nn.Softmax(dim=1)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, img_feat, txt_feat):
        # Project to common space
        img_proj = self.relu(self.img_proj(img_feat))
        txt_proj = self.relu(self.txt_proj(txt_feat))
        
        # Compute attention weights
        concat = torch.cat([img_proj, txt_proj], dim=1)
        attention_weights = self.attention(concat)  # [batch, 2]
        
        # Apply attention weights
        img_weighted = img_proj * attention_weights[:, 0:1]
        txt_weighted = txt_proj * attention_weights[:, 1:2]
        
        # Fuse features
        fused = img_weighted + txt_weighted
        fused = self.dropout(fused)
        
        return fused, attention_weights


# ============================================
# GATED FUSION MODEL (Alternative)
# ============================================
class GatedFusion(nn.Module):
    """
    Gated fusion mechanism - learns a gate to balance modalities
    """
    
    def __init__(self, img_dim=IMG_FEATURE_DIM, txt_dim=TEXT_FEATURE_DIM, fusion_dim=FUSION_DIM):
        super(GatedFusion, self).__init__()
        
        self.img_fc = nn.Linear(img_dim, fusion_dim)
        self.txt_fc = nn.Linear(txt_dim, fusion_dim)
        
        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, img_feat, txt_feat):
        hi = self.relu(self.img_fc(img_feat))
        ht = self.relu(self.txt_fc(txt_feat))
        
        # Compute gate
        concat = torch.cat([hi, ht], dim=1)
        z = self.gate(concat)
        
        # Gated fusion
        fused = z * hi + (1 - z) * ht
        fused = self.dropout(fused)
        
        return fused, z


# ============================================
# MULTIMODAL CLASSIFIER
# ============================================
class MultimodalClassifier(nn.Module):
    """
    Classifier that takes fused features and organ type
    Predicts disease presence/absence
    """
    
    def __init__(self, fusion_dim=FUSION_DIM, num_organs=NUM_ORGANS, hidden_dim=CLASSIFIER_HIDDEN):
        super(MultimodalClassifier, self).__init__()
        
        # Organ embedding to handle different organ types
        self.organ_embed = nn.Embedding(num_organs, 64)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features, organ_indices):
        # Get organ embeddings
        organ_emb = self.organ_embed(organ_indices)
        
        # Concatenate with fused features
        combined = torch.cat([fused_features, organ_emb], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output


# ============================================
# TRAINING FUNCTIONS
# ============================================
class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, fusion_type='attention'):
        self.fusion_type = fusion_type
        
        # Initialize models
        if fusion_type == 'attention':
            self.fusion_model = AttentionFusion().to(DEVICE)
        else:
            self.fusion_model = GatedFusion().to(DEVICE)
        
        self.classifier = MultimodalClassifier().to(DEVICE)
        
        # Optimizers
        self.fusion_optimizer = optim.AdamW(
            self.fusion_model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        self.classifier_optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.fusion_model.train()
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            img_feat = batch['img_feat'].to(DEVICE)
            txt_feat = batch['txt_feat'].to(DEVICE)
            labels = batch['label'].to(DEVICE).unsqueeze(1)
            organ_idx = batch['organ_idx'].to(DEVICE)
            
            # Zero gradients
            self.fusion_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            
            # Forward pass
            fused, _ = self.fusion_model(img_feat, txt_feat)
            predictions = self.classifier(fused, organ_idx)
            
            # Compute loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.fusion_optimizer.step()
            self.classifier_optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted_labels = (predictions > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate model"""
        self.fusion_model.eval()
        self.classifier.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                img_feat = batch['img_feat'].to(DEVICE)
                txt_feat = batch['txt_feat'].to(DEVICE)
                labels = batch['label'].to(DEVICE).unsqueeze(1)
                organ_idx = batch['organ_idx'].to(DEVICE)
                
                # Forward pass
                fused, _ = self.fusion_model(img_feat, txt_feat)
                predictions = self.classifier(fused, organ_idx)
                
                # Compute loss
                loss = self.criterion(predictions, labels)
                
                # Statistics
                total_loss += loss.item()
                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS):
        """Full training loop"""
        logger.info("Starting training...")
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_models('best')
                logger.info(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        logger.info(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    
    def save_models(self, prefix='final'):
        """Save trained models"""
        torch.save(self.fusion_model.state_dict(), 
                  f"{MODELS_DIR}/{prefix}_fusion.pth")
        torch.save(self.classifier.state_dict(),
                  f"{MODELS_DIR}/{prefix}_classifier.pth")
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{MODELS_DIR}/training_history.png")
        logger.info(f"Training history plot saved to {MODELS_DIR}/training_history.png")


def main():
    """Main training function"""
    logger.info("=" * 50)
    logger.info("Model Training")
    logger.info("=" * 50)
    
    # Load datasets
    train_dataset = MultimodalDataset(f"{PROJECT_ROOT}/train_features.npz")
    val_dataset = MultimodalDataset(f"{PROJECT_ROOT}/val_features.npz")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Initialize trainer
    trainer = ModelTrainer(fusion_type='attention')
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_models('final')
    
    # Plot training history
    trainer.plot_history()
    
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()