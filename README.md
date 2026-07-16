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

The dataset is downloaded automatically on first run via `kagglehub`.
Make sure you have a Kaggle account and your API credentials configured:

```bash
# Place kaggle.json in ~/.kaggle/ (download from Kaggle → Settings → API)
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

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
