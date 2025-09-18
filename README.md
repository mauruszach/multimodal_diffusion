# multimodal diffusion

A multimodal diffusion model for cross-modal generation between smell, image, and audio modalities.

## Project Structure

- `configs/`: Model configuration files
- `data/`: Raw and preprocessed data
  - `smells/`: E-nose traces (npz/h5)
  - `images/`: Frames or webdataset shards
  - `audio/`: Wav/encoded latents
- `models/`: Model implementations
  - `modules/`: Core model components
  - `heads/`: Output heads for different tasks
- `train/`: Training code and utilities
- `sample/`: Sampling and inference code
- `scripts/`: Data preprocessing scripts
- `tests/`: Test cases

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train/train_mvp.py --config configs/mvp.yaml
```

### Sampling
```bash
python sample/sample_any2any.py --config configs/mvp.yaml --input_modality smell --output_modalities image,audio
```

## License
[MIT](LICENSE)
