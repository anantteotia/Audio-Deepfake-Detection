"""
Training script for audio deepfake detection models.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.data_loader import AudioDeepfakeDataset
from models.baseline import get_model as get_baseline_model
from models.transformer import get_transformer_model


def train_epoch(model, dataloader, criterion, optimizer, device, is_transformer=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if is_transformer:
            # For transformers, expect raw waveform
            logits = model(features.squeeze(1))
        else:
            logits = model(features)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, is_transformer=False):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            
            if is_transformer:
                logits = model(features.squeeze(1))
            else:
                logits = model(features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def train(args):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )
    
    # Load data
    train_dataset = AudioDeepfakeDataset(
        data_dir=os.path.join(args.data_dir, "train"),
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )
    
    val_dataset = AudioDeepfakeDataset(
        data_dir=os.path.join(args.data_dir, "val"),
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    if args.model_type == "transformer":
        model = get_transformer_model(args.model_name, num_classes=2)
        is_transformer = True
    else:
        model = get_baseline_model(args.model_name, num_classes=2)
        is_transformer = False
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, is_transformer
        )
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, is_transformer)
        
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy'],
                "val_f1": val_metrics['f1'],
                "val_auc": val_metrics['auc'],
                "epoch": epoch
            })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--feature_type", type=str, default="mel", choices=["raw", "mel", "mfcc"])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_duration", type=float, default=4.0)
    
    # Model
    parser.add_argument("--model_type", type=str, default="transformer", choices=["baseline", "transformer"])
    parser.add_argument("--model_name", type=str, default="wav2vec2")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="audio-deepfake")
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)