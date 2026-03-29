# Data Directory

Place your audio deepfake dataset here with the following structure:

```
data/
├── train/
│   ├── real/
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   └── fake/
│       ├── sample1.wav
│       ├── sample2.wav
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Expected Format

- **Format**: WAV (16-bit PCM)
- **Sample Rate**: 16 kHz
- **Duration**: 1-10 seconds (will be padded/truncated to max_duration)
- **Channels**: Mono

## Downloading Datasets

### ASVspoof 2019/2021
```bash
# Visit https://www.asvspoof.org/
# Download the LA (logical access) dataset
```

### WaveFake
```bash
# Visit https://github.com/JackieTai/WaveFake
# Follow instructions to download
```

### FakeAVCeleb
```bash
# Visit https://github.com/DashanTi/FakeAVCeleb
```

## Quick Stats

Run to get dataset statistics:
```python
from src.data_loader import get_dataset_stats
stats = get_dataset_stats("data/train")
print(stats)
```