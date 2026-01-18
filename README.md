# Audio Deepfake Detection using Transformer-based Models

## Project Overview

This project implements machine learning models to detect AI-generated or manipulated audio (deepfakes) using state-of-the-art transformer architectures. With the proliferation of voice cloning and synthetic speech generation technologies, detecting fake audio has become crucial for applications in cybersecurity, journalism verification, legal evidence authentication, and fraud prevention.

## Objectives

- Implement and fine-tune transformer-based models for audio deepfake detection
- Compare performance across different architectures
- Analyze attention patterns to understand acoustic features that distinguish real from synthetic audio
- Evaluate computational efficiency and accuracy trade-offs

## Models to Implement

### Primary Models
- **Wav2Vec 2.0**: Self-supervised learning model for speech representation
- **HuBERT** (Hidden-Unit BERT): Masked prediction approach for speech processing
- **Audio Spectrogram Transformer (AST)**: Vision transformer adapted for audio spectrograms

### Baseline Models (for comparison)
- Traditional ML approaches (SVM, Random Forest)
- CNN-based architectures

## Datasets

### Primary Datasets
- **ASVspoof 2019/2021**: Standard benchmark for audio anti-spoofing
- **FakeAVCeleb**: Audio-visual deepfake dataset
- **WaveFake**: Synthetic speech detection dataset

### Data Characteristics
- Real human speech samples
- AI-generated audio from various synthesis methods (TTS, voice conversion, vocoder artifacts)
- Multiple languages and speakers
- Various recording conditions

## Technical Approach

### 1. Data Preprocessing
- Audio loading and normalization
- Feature extraction:
  - Raw waveforms (for Wav2Vec 2.0, HuBERT)
  - Mel spectrograms (for AST)
  - MFCCs (for baseline models)
- Data augmentation (time stretching, pitch shifting, noise injection)

### 2. Model Implementation
- Load pre-trained transformer models
- Fine-tune on deepfake detection task
- Implement attention visualization
- Compare transfer learning vs. training from scratch

### 3. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Equal Error Rate (EER)
- ROC-AUC
- Confusion matrices
- Computational efficiency (inference time, model size)

### 4. Analysis
- Attention pattern visualization
- Feature importance analysis
- Error analysis (false positives/negatives)
- Generalization across different synthesis methods

## Project Timeline

- **Week 1-2**: Literature review, dataset preparation
- **Week 3-4**: Baseline model implementation
- **Week 5-6**: Transformer model implementation and fine-tuning
- **Week 7-8**: Evaluation, analysis, and documentation

## Technology Stack

- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Audio Processing**: librosa, torchaudio, soundfile
- **Pre-trained Models**: Hugging Face Transformers
- **Visualization**: matplotlib, seaborn, tensorboard
- **Experiment Tracking**: Weights & Biases (wandb)

## Research Alignment

This project aligns with ongoing research at Dr. Li Yang's lab (UNC Charlotte) on optimizing Audio and Video Transformers, specifically exploring attention-based mechanisms and computational efficiency in multimodal models.

## Expected Outcomes

1. Trained deepfake detection models with competitive performance
2. Comparative analysis of transformer architectures for audio classification
3. Insights into acoustic features that distinguish real from synthetic audio
4. Potential for computational optimization in detection pipeline
5. Comprehensive documentation and reproducible code

## References

Key papers to review:
1. ASVspoof Challenge papers (2019, 2021)
2. "Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
3. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction"
4. "Audio Spectrogram Transformer for Audio Classification"
5. Recent deepfake detection papers from ICASSP, Interspeech, IEEE conferences

## Status

🚧 **Project in planning phase** - Repository setup complete, beginning literature review and dataset acquisition.

## Contact

**Anant Teotia**  
Graduate Research Assistant, UNC Charlotte  
Email: anant.teotia@outlook.com

## License

MIT License (to be added)
