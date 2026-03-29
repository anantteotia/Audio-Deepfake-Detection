"""
Transformer-based models for audio deepfake detection.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel, AutoModel
from typing import Optional


class Wav2Vec2Classifier(nn.Module):
    """Wav2Vec 2.0 for audio deepfake detection."""
    
    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_features: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained Wav2Vec 2.0
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        # Freeze if specified
        if freeze_features:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.wav2vec.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            waveform: Raw audio waveform [B, T]
            attention_mask: Attention mask for variable length audio
        """
        # Get Wav2Vec outputs
        outputs = self.wav2vec(
            waveform, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        
        # Mean pooling over time
        pooled = hidden_states.mean(dim=1)  # [B, H]
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class HubertClassifier(nn.Module):
    """HuBERT for audio deepfake detection."""
    
    def __init__(
        self,
        pretrained_model: str = "facebook/hubert-base-ls960",
        num_classes: int = 2,
        freeze_features: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Load pretrained HuBERT
        self.hubert = HubertModel.from_pretrained(pretrained_model)
        
        # Freeze if specified
        if freeze_features:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Get hidden size
        hidden_size = self.hubert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass."""
        outputs = self.hubert(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits


class ASTClassifier(nn.Module):
    """Audio Spectrogram Transformer for deepfake detection."""
    
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Patch embedding for spectrograms
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Conv2d for patch extraction (treat spectrogram as image)
        self.patch_embed = nn.Conv2d(
            1, embed_dim, 
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Spectrogram [B, 1, F, T]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use cls token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple transformer models."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
    
    def forward(self, waveform: torch.Tensor):
        """Average predictions from all models."""
        logits_list = []
        
        for model in self.models:
            with torch.no_grad():  # Freeze backbone
                logits = model(waveform)
                logits_list.append(logits)
        
        # Weighted average
        logits_stack = torch.stack(logits_list)
        weighted_logits = (logits_stack * self.weights.view(-1, 1, 1)).sum(dim=0)
        
        return weighted_logits


def get_transformer_model(model_name: str, num_classes: int = 2):
    """Factory function to get transformer models."""
    
    models = {
        "wav2vec2": lambda: Wav2Vec2Classifier(
            pretrained_model="facebook/wav2vec2-base",
            num_classes=num_classes
        ),
        "wav2vec2-large": lambda: Wav2Vec2Classifier(
            pretrained_model="facebook/wav2vec2-large-lv60",
            num_classes=num_classes
        ),
        "hubert": lambda: HubertClassifier(
            pretrained_model="facebook/hubert-base-ls960",
            num_classes=num_classes
        ),
        "ast": lambda: ASTClassifier(num_classes=num_classes),
    }
    
    if model_name.lower() in models:
        return models[model_name.lower()]()
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")


if __name__ == "__main__":
    print("Transformer Models")
    print("=" * 40)
    print("Available models:")
    print("  - Wav2Vec2Classifier: facebook/wav2vec2-base")
    print("  - Wav2Vec2Classifier (large): facebook/wav2vec2-large-lv60")
    print("  - HubertClassifier: facebook/hubert-base-ls960")
    print("  - ASTClassifier: Audio Spectrogram Transformer")
    print("\nFactory: get_transformer_model('wav2vec2')")