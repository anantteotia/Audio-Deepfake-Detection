"""
Quick training test - runs 2 epochs to verify pipeline works.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.data_loader import AudioDeepfakeDataset
from models.baseline import AudioClassifierCNN

# Config
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = str(REPO_ROOT / "data")
OUTPUT_DIR = str(REPO_ROOT / "checkpoints")
BATCH_SIZE = 16
EPOCHS = 2
LR = 1e-3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data
train_ds = AudioDeepfakeDataset(
    data_dir=os.path.join(DATA_DIR, 'train'),
    feature_type='mel', sample_rate=16000, max_duration=2.0
)
val_ds = AudioDeepfakeDataset(
    data_dir=os.path.join(DATA_DIR, 'val'),
    feature_type='mel', sample_rate=16000, max_duration=2.0
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# Model
model = AudioClassifierCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
best_val_acc = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []
    
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(logits.argmax(dim=1).cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    train_acc = accuracy_score(train_labels, train_preds)
    print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
    
    # Validate
    model.eval()
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1} - Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
        print(f"  -> New best model saved!")

print(f"\nTraining complete! Best Val Acc: {best_val_acc:.4f}")