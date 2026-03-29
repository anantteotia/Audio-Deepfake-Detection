"""
Dataset download utilities.
"""

import os
import urllib.request
import zipfile
import gdown
from pathlib import Path


def download_asvspoof(data_dir: str, year: int = 2019):
    """
    Download ASVspoof dataset.
    
    Note: You need to register at https://www.asvspoof.org/ to access the data.
    """
    print("ASVspoof dataset requires registration at https://www.asvspoof.org/")
    print("Please download manually and place in data directory.")
    
    # Create placeholder
    os.makedirs(os.path.join(data_dir, "asvspoof"), exist_ok=True)
    print(f"Created directory: {data_dir}/asvspoof")


def download_wavefake(data_dir: str):
    """Download WaveFake dataset from Google Drive."""
    output_dir = os.path.join(data_dir, "wavefake")
    os.makedirs(output_dir, exist_ok=True)
    
    # WaveFake Google Drive IDs (check latest from official repo)
    # This is a placeholder - check https://github.com/JackieTai/WaveFake for actual links
    print("WaveFake: Please check https://github.com/JackieTai/WaveFake for download links")
    print(f"Place downloaded files in: {output_dir}")


def download_fakeavceleb(data_dir: str):
    """Download FakeAVCeleb dataset."""
    output_dir = os.path.join(data_dir, "fakeavceleb")
    os.makedirs(output_dir, exist_ok=True)
    
    print("FakeAVCeleb: Please check https://github.com/DashanTi/FakeAVCeleb for download links")
    print(f"Place downloaded files in: {output_dir}")


def download_sample_data(data_dir: str, dataset: str = "mini"):
    """
    Download sample/mini dataset for testing.
    
    Args:
        data_dir: Directory to save data
        dataset: 'mini' for small test set
    """
    output_dir = Path(data_dir)
    
    if dataset == "mini":
        # Create mini synthetic dataset for testing
        print("Creating mini synthetic dataset for testing...")
        
        for split in ["train", "val", "test"]:
            for label in ["real", "fake"]:
                dir_path = output_dir / split / label
                dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Created directory structure in: {output_dir}")
        print("Note: No actual audio files - you'll need to add your own or download a dataset")


def prepare_data(data_dir: str):
    """
    Prepare data directory with proper structure.
    
    Args:
        data_dir: Root data directory
    """
    data_path = Path(data_dir)
    
    splits = ["train", "val", "test"]
    labels = ["real", "fake"]
    
    for split in splits:
        for label in labels:
            dir_path = data_path / split / label
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Data directory structure created:")
    print(f"  {data_dir}/")
    for split in splits:
        for label in labels:
            print(f"  ├── {split}/{label}/")
    
    print("\nNext steps:")
    print("1. Download a dataset (ASVspoof, WaveFake, FakeAVCeleb)")
    print("2. Organize files in the structure above")
    print("3. Run training: python src/train.py --data_dir data")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare audio deepfake datasets")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to download")
    
    args = parser.parse_args()
    
    # Create directory structure
    prepare_data(args.data_dir)
    
    # Download specific dataset if requested
    if args.dataset:
        if args.dataset == "asvspoof":
            download_asvspoof(args.data_dir)
        elif args.dataset == "wavefake":
            download_wavefake(args.data_dir)
        elif args.dataset == "fakeavceleb":
            download_fakeavceleb(args.data_dir)
        elif args.dataset == "mini":
            download_sample_data(args.data_dir)
        else:
            print(f"Unknown dataset: {args.dataset}")