# Audio Deepfake Detection using Transformer-based Models

## Project Overview

This project implements machine learning models to detect AI-generated or manipulated audio (deepfakes) using state-of-the-art transformer architectures. With the proliferation of voice cloning and synthetic speech generation technologies, detecting fake audio has become crucial for applications in cybersecurity, journalism verification, legal evidence authentication, and fraud prevention.

## Project Status

🚀 **Implementation Started** - Core modules complete, ready for dataset and training.

## Quick Start

```bash
# Clone and setup
cd G:\Code\Audio-Deepfake-Detection
pip install -r requirements.txt

# Prepare data directory structure
python src/download.py --data_dir data

# Download a dataset (ASVspoof, WaveFake, FakeAVCeleb)
# Or place your own data in data/train/real and data/train/fake

# Train a model
python src/train.py --model_type transformer --model_name wav2vec2 --epochs 10

# Evaluate
python src/evaluate.py --data_dir data/test --model_path checkpoints/best_model.pt
```

## Project Structure

```
Audio-Deepfake-Detection/
├── data/                    # Dataset directory
│   ├── train/               # Training data
│   │   ├── real/            # Real audio files
│   │   └── fake/            # Fake audio files
│   ├── val/                 # Validation data
│   └── test/                # Test data
├── src/                     # Source code
│   ├── data_loader.py       # Data loading & preprocessing
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── download.py          # Dataset download utilities
├── models/                  # Model implementations
│   ├── baseline.py          # CNN, MLP, Random Forest, SVM
│   └── transformer.py       # Wav2Vec2, HuBERT, AST
├── configs/                 # Configuration files
│   └── default.yaml         # Default training config
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Python dependencies
└── README.md
```

## Models Implemented

### Transformer Models
| Model | Description | Pretrained |
|-------|-------------|------------|
| Wav2Vec2 | Self-supervised speech representation | facebook/wav2vec2-base |
| HuBERT | Masked prediction for speech | facebook/hubert-base-ls960 |
| AST | Vision transformer for spectrograms |从头训练 |

### Baseline Models
| Model | Type |
|-------|------|
| CNN | Convolutional Neural Network |
| MLP | Multi-layer Perceptron |
| Random Forest | Traditional ML |
| SVM | Support Vector Machine |

## Datasets

### Supported Datasets
- **ASVspoof 2019/2021**: Standard benchmark for audio anti-spoofing
- **FakeAVCeleb**: Audio-visual deepfake dataset
- **WaveFake**: Synthetic speech detection dataset

### Data Format
- Format: WAV (16-bit PCM)
- Sample Rate: 16 kHz
- Duration: 1-10 seconds (padded/truncated to max_duration)
- Channels: Mono

## Training

### Basic Training
```bash
python src/train.py --data_dir data --epochs 10 --batch_size 16 --lr 1e-4
```

### With Weights & Biases
```bash
python src/train.py --data_dir data --use_wandb --wandb_project audio-deepfake --run_name exp1
```

### Training Options
```bash
python src/train.py --help
# --model_type [transformer|baseline]
# --model_name [wav2vec2|hubert|ast|cnn|mlp]
# --feature_type [raw|mel|mfcc]
# --epochs, --batch_size, --lr, etc.
```

## Evaluation

```bash
python src/evaluate.py \
    --data_dir data/test \
    --model_path checkpoints/best_model.pt \
    --model_type transformer \
    --model_name wav2vec2 \
    --output_path results.json
```

Outputs: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix

## Configuration

Edit `configs/default.yaml` or create your own config:

```yaml
model:
  model_type: "transformer"
  model_name: "wav2vec2"
  freeze_features: true

training:
  epochs: 10
  lr: 0.0001

logging:
  use_wandb: true
  wandb_project: "audio-deepfake"
```

## Technical Approach

### 1. Data Preprocessing
- Audio loading with librosa
- Feature extraction (raw, mel spectrogram, MFCC)
- Padding/truncation to fixed length
- Data augmentation (optional)

### 2. Model Implementation
- Load pretrained transformers from Hugging Face
- Fine-tune on deepfake detection task
- Optional: freeze backbone, train only classifier

### 3. Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Equal Error Rate (EER)
- Confusion matrices

## Technology Stack

- **Python**: 3.8+
- **PyTorch**: Deep learning framework
- **librosa**: Audio processing
- **transformers**: Hugging Face pretrained models
- **wandb**: Experiment tracking
- **sklearn**: Baseline models & metrics

## Research Alignment

This project aligns with ongoing research at Dr. Li Yang's lab (UNC Charlotte) on optimizing Audio and Video Transformers, specifically exploring attention-based mechanisms and computational efficiency in multimodal models.

## To Do

- [ ] Download and prepare dataset
- [ ] Train baseline models (CNN, Random Forest)
- [ ] Train transformer models (Wav2Vec2, HuBERT)
- [ ] Compare model performance
- [ ] Analyze attention patterns
- [ ] Optimize for efficiency
- [ ] Write final report

## License

MIT License

## Contact

**Anant Teotia**  
Graduate Research Assistant, UNC Charlotte  
Email: anant.teotia@outlook.com