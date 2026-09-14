# pikacnn

Pokemon image classification with a **JAX/Flax** convolutional neural network.
Trained on the 1025 species of the National Dex, generations I to IX.

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

The dataset is versioned directly in this repo under
[`data/pokemon/`](data/pokemon) — one folder per class, 128×128 JPEGs, all
**1025 species** of the National Dex (generations I–IX). No Kaggle account
needed: just clone and train.

```
data/pokemon/
├── Bulbasaur/
├── Charmander/
├── ...            (1025 class folders)
└── Zygarde/
```

Generation I uses rendered images (from the Kaggle *Pokemon Generation One*
mirrors); generations II–IX use official sprites (no shiny variants). Sources
and licenses: `Dusduo/1stGen-Pokemon-Images` and
`RogerKoala/gen1-pokemon-images` (MIT) for gen I,
`JJMack/pokemon-classification-gen1-9` (CC-BY-NC-SA-4.0) for gens II–IX.

For higher-resolution training on generation I you can point `train.py` at
the full Kaggle [Pokemon Generation One](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one)
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
