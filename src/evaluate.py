"""
Evaluation script for audio deepfake detection models.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.data_loader import AudioDeepfakeDataset
from models.baseline import get_model as get_baseline_model
from models.transformer import get_transformer_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def load_model(model_path: str, model_type: str, model_name: str, device: str):
    """Load a trained model."""
    if model_type == "transformer":
        model = get_transformer_model(model_name, num_classes=2)
    else:
        model = get_baseline_model(model_name, num_classes=2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, dataloader, device, is_transformer=False):
    """Evaluate model on a dataset."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            
            if is_transformer:
                logits = model(features.squeeze(1))
            else:
                logits = model(features)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    return metrics, all_preds, all_labels, all_probs


def main(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = AudioDeepfakeDataset(
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load model
    model = load_model(
        args.model_path,
        args.model_type,
        args.model_name,
        device
    )
    print(f"Loaded model from: {args.model_path}")
    
    is_transformer = args.model_type == "transformer"
    
    # Evaluate
    metrics, preds, labels, probs = evaluate(model, dataloader, device, is_transformer)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC-ROC:  {metrics['auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  Predicted:  0      1")
    print(f"  Actual 0:  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"  Actual 1:  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Fake", "Real"]))
    
    # Save metrics
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake detection model")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["baseline", "transformer"])
    parser.add_argument("--model_name", type=str, default="wav2vec2")
    parser.add_argument("--feature_type", type=str, default="mel")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_duration", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_path", type=str, default=None, help="Path to save metrics JSON")
    
    args = parser.parse_args()
    main(args)