# pikacnn

Pokemon image classification with a **JAX/Flax** convolutional neural network.
Trained on the [Pokemon Generation One](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) dataset.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> For GPU support, replace `jax[cpu]` with `jax[cuda12]` in `requirements.txt`.

## Dataset

The first-generation dataset (151 species, 128×128 JPEGs) is versioned directly
in this repo under [`data/pokemon/`](data/pokemon) — one folder per class. No
Kaggle account needed: just clone and train.

```
data/pokemon/
├── Bulbasaur/
├── Charmander/
├── ...            (151 class folders)
└── Zubat/
```

For higher-resolution training you can instead point `train.py` at the full
Kaggle [Pokemon Generation One](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one)
set (~800 imgs/species): delete or rename `data/pokemon` and configure Kaggle
API credentials (`kagglehub` will download it automatically on first run).

## Training

```bash
python train.py
```

All hyperparameters are in the `CONFIG` dict at the top of `train.py`.

## Architecture

| Layer          | Details                          |
|----------------|----------------------------------|
| Conv2D ×5     | 32→64→128→256→512 filters, 3×3  |
| BatchNorm      | After each conv                  |
| MaxPooling     | 2×2 after each block             |
| Dense          | 1024 units, ReLU                 |
| Dropout        | 0.5                              |
| Output         | Dense(num_classes), softmax      |

Optimizer: **AdamW** with cosine decay + warmup.

## License

MIT
