"""
Baseline ML models for audio deepfake detection.
"""

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional


class AudioClassifierMLP(nn.Module):
    """Simple MLP for audio classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.model(x)


class AudioClassifierCNN(nn.Module):
    """CNN for spectrogram-based audio classification."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BaselineModel:
    """Wrapper for traditional ML baselines."""
    
    def __init__(self, model_type: str = "rf", scale: bool = True):
        """
        Args:
            model_type: 'rf' for Random Forest, 'svm' for SVM
            scale: Whether to scale features
        """
        self.model_type = model_type
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.model = None
        self._build_model()
    
    def _build_model(self):
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == "svm":
            self.model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Handle 3D input (for spectrograms) - flatten
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray):
        """Predict labels."""
        if self.scaler:
            X = self.scaler.transform(X)
        
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray):
        """Predict probabilities."""
        if self.scaler:
            X = self.scaler.transform(X)
        
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict_proba(X)


def get_model(model_name: str, num_classes: int = 2):
    """Factory function to get models."""
    
    models = {
        "mlp": lambda: AudioClassifierMLP(input_dim=64000, hidden_dim=512, num_classes=num_classes),
        "cnn": lambda: AudioClassifierCNN(num_classes=num_classes),
    }
    
    if model_name.lower() in models:
        return models[model_name.lower()]()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")


if __name__ == "__main__":
    print("Baseline Models")
    print("=" * 40)
    print("PyTorch Models:")
    print("  - AudioClassifierMLP: Simple feedforward network")
    print("  - AudioClassifierCNN: CNN for spectrograms")
    print("\nSklearn Models:")
    print("  - BaselineModel with 'rf' (Random Forest)")
    print("  - BaselineModel with 'svm' (SVM)")
    print("\nFactory: get_model('mlp') or get_model('cnn')")