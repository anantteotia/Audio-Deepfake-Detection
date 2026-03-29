"""
Data loading and preprocessing for audio deepfake detection.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):
    """Dataset for audio deepfake detection."""
    
    def __init__(
        self,
        data_dir: str,
        feature_type: str = "mel",
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        transform=None,
        is_train: bool = True
    ):
        """
        Args:
            data_dir: Directory containing audio files
            feature_type: 'mel', 'mfcc', or 'raw'
            sample_rate: Target sample rate
            max_duration: Max audio duration in seconds
            transform: Optional transform to apply
            is_train: Train or eval mode
        """
        self.data_dir = Path(data_dir)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform
        self.is_train = is_train
        
        # Load metadata
        self.samples = self._load_metadata()
    
    def _load_metadata(self) -> List[Tuple[str, int]]:
        """Load file paths and labels from data directory."""
        samples = []
        
        # Expected structure: data_dir/real/*.wav, data_dir/fake/*.wav
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"
        
        if real_dir.exists():
            for f in real_dir.glob("*.wav"):
                samples.append((str(f), 1))  # 1 = real
        
        if fake_dir.exists():
            for f in fake_dir.glob("*.wav"):
                samples.append((str(f), 0))  # 0 = fake
        
        return samples
    
    def _load_audio(self, path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        # Load audio
        waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        
        # Pad or truncate
        if len(waveform) < self.max_samples:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
        else:
            waveform = waveform[:self.max_samples]
        
        return waveform
    
    def _extract_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract features based on feature_type."""
        if self.feature_type == "raw":
            return waveform
        
        elif self.feature_type == "mel":
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, 
                sr=self.sample_rate,
                n_mels=64,
                hop_length=160
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel
        
        elif self.feature_type == "mfcc":
            # MFCC
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=40
            )
            return mfcc
        
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        # Load and process
        waveform = self._load_audio(path)
        features = self._extract_features(waveform)
        
        # Convert to tensor
        if self.feature_type == "raw":
            features = torch.from_numpy(features).float()
        else:
            features = torch.from_numpy(features).unsqueeze(0).float()
        
        if self.transform:
            features = self.transform(features)
        
        return features, label


def get_dataset_stats(data_dir: str) -> dict:
    """Get dataset statistics."""
    dataset = AudioDeepfakeDataset(data_dir)
    
    real_count = sum(1 for _, label in dataset.samples if label == 1)
    fake_count = sum(1 for _, label in dataset.samples if label == 0)
    
    return {
        "total": len(dataset.samples),
        "real": real_count,
        "fake": fake_count,
        "real_ratio": real_count / len(dataset.samples) if dataset.samples else 0,
    }


if __name__ == "__main__":
    # Quick test
    print("Audio Deepfake Detection Data Loader")
    print("=" * 40)
    print("Available functions:")
    print("  - AudioDeepfakeDataset: PyTorch Dataset class")
    print("  - get_dataset_stats(): Get dataset statistics")