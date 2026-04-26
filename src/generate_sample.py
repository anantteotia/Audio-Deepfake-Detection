"""
Generate synthetic audio samples for testing the pipeline.
This creates simple test audio files to verify data loading works.
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa


def generate_sine_wave(frequency, duration, sample_rate=16000):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def generate_noise(duration, sample_rate=16000):
    """Generate white noise."""
    return np.random.randn(int(sample_rate * duration)) * 0.1


def generate_sample(output_path, sample_type="real", sample_rate=16000, duration=2.0):
    """Generate a single audio sample."""
    if sample_type == "real":
        # Real audio: clean sine wave with some harmonics
        freq = np.random.randint(100, 400)
        audio = generate_sine_wave(freq, duration, sample_rate)
        # Add slight harmonics
        audio += 0.3 * generate_sine_wave(freq * 2, duration, sample_rate)
        audio += 0.1 * generate_sine_wave(freq * 3, duration, sample_rate)
        # Add very subtle noise
        audio += np.random.randn(int(sample_rate * duration)) * 0.02
    else:
        # Fake audio: more noise, artifacts, distortion
        freq = np.random.randint(100, 400)
        audio = generate_sine_wave(freq, duration, sample_rate)
        # Add more harmonics (sounds unnatural)
        audio += 0.6 * generate_sine_wave(freq * 2, duration, sample_rate)
        audio += 0.5 * generate_sine_wave(freq * 3, duration, sample_rate)
        audio += 0.4 * generate_sine_wave(freq * 1.5, duration, sample_rate)  # Dissonant
        # Add noticeable noise
        audio += np.random.randn(int(sample_rate * duration)) * 0.15
        # Add clicks/pops (artifact simulation)
        click_samples = np.random.randint(0, int(sample_rate * duration), 5)
        for cs in click_samples:
            end = min(cs + 50, audio.shape[0])
            audio[cs:end] += np.random.randn(end - cs) * 0.5
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save
    sf.write(output_path, audio, sample_rate)


def generate_dataset(base_dir, num_samples=50, sample_rate=16000, duration=2.0):
    """Generate complete synthetic dataset."""
    base_path = Path(base_dir)
    
    splits = {
        "train": {"real": int(num_samples * 0.7), "fake": int(num_samples * 0.7)},
        "val": {"real": int(num_samples * 0.15), "fake": int(num_samples * 0.15)},
        "test": {"real": int(num_samples * 0.15), "fake": int(num_samples * 0.15)},
    }
    
    total_real = 0
    total_fake = 0
    
    for split, counts in splits.items():
        for label, count in counts.items():
            dir_path = base_path / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                filename = f"{label}_{i:04d}.wav"
                filepath = dir_path / filename
                generate_sample(str(filepath), label, sample_rate, duration)
                
                if label == "real":
                    total_real += 1
                else:
                    total_fake += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i+1}/{count} {split}/{label} samples")
    
    print(f"\nDataset created successfully!")
    print(f"Total samples: {total_real + total_fake}")
    print(f"  - Real: {total_real}")
    print(f"  - Fake: {total_fake}")
    print(f"Location: {base_path}")


def verify_dataset(data_dir):
    """Verify dataset structure and can be loaded."""
    print("\nVerifying dataset...")
    
    from src.data_loader import AudioDeepfakeDataset
    
    try:
        # Try loading train split (expected structure: data/train/{real,fake})
        dataset = AudioDeepfakeDataset(
            data_dir=os.path.join(data_dir, "train"),
            feature_type="mel",
            sample_rate=16000,
            max_duration=4.0
        )
        
        print("OK: Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        
        # Get a sample
        features, label = dataset[0]
        print(f"  Sample shape: {features.shape}")
        print(f"  Sample label: {label}")
        
        # Count classes
        real_count = sum(1 for _, l in dataset.samples if l == 1)
        fake_count = sum(1 for _, l in dataset.samples if l == 0)
        print(f"  Real: {real_count}, Fake: {fake_count}")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--num_samples", type=int, default=60, help="Samples per class (real/fake) total, then split)")
    parser.add_argument("--verify", action="store_true", help="Verify after generation")
    
    args = parser.parse_args()
    
    # Generate dataset
    print("Generating synthetic audio dataset...")
    print("=" * 50)
    generate_dataset(args.data_dir, num_samples=args.num_samples)
    
    if args.verify:
        verify_dataset(args.data_dir)