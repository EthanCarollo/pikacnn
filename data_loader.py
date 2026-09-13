"""
Data loader for the Pokemon image dataset (National Dex #001-#1025).

The 151-species gen-1 dataset lives in `data/pokemon` (one folder per class,
128x128 JPEGs) and is loaded directly from there. If it is missing, `train.py`
falls back to downloading the full-resolution Kaggle copy via kagglehub.
"""

import os
import numpy as np
from PIL import Image


LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pokemon")


def download_dataset() -> str:
    """Download Pokemon Generation One from Kaggle. Returns the local path."""
    import kagglehub

    print("[data] dataset/pokemon not found, downloading from kagglehub ...")
    path = kagglehub.dataset_download("thedagger/pokemon-generation-one")
    print(f"[data] dataset downloaded to: {path}")
    return path


def get_data_dir() -> str:
    """Return the committed dataset if present, otherwise download it."""
    if os.path.isdir(LOCAL_DATA_DIR):
        return LOCAL_DATA_DIR
    return download_dataset()


def load_dataset(data_dir: str, image_size: tuple[int, int] = (128, 128)):
    """Walk through class subdirectories and load all images as numpy arrays.

    Args:
        data_dir: root directory containing one subfolder per class.
        image_size: target (height, width) for resizing.

    Returns:
        images:  float32 array of shape (N, H, W, 3), values in [0, 1].
        labels:  int64 array of shape (N,).
        class_names: list of class folder names, sorted alphabetically.
    """
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    if not class_names:
        raise FileNotFoundError(
            f"[data] no class subdirectories found in {data_dir}. "
            f"Is the dataset structured as data_dir/class_name/*.png?"
        )

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"[data] found {len(class_names)} classes")

    images: list[np.ndarray] = []
    labels: list[int] = []
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if os.path.splitext(fname)[1].lower() in valid_extensions:
                img_path = os.path.join(class_dir, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(image_size[::-1], Image.LANCZOS)  # (W, H)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(class_to_idx[class_name])
                except Exception:
                    print(f"[data] warning: skipped corrupt image {img_path}")

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    print(f"[data] loaded {len(images_arr)} images — shape {images_arr.shape}")
    return images_arr, labels_arr, class_names


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Shuffle and split images/labels into train, val and test sets.

    Returns a dict with keys 'train', 'val', 'test', each holding a
    tuple (images, labels).
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    indices = rng.permutation(n)

    test_end = int(n * test_ratio)
    val_end = test_end + int(n * val_ratio)

    test_idx = indices[:test_end]
    val_idx = indices[test_end:val_end]
    train_idx = indices[val_end:]

    splits = {
        "train": (images[train_idx], labels[train_idx]),
        "val": (images[val_idx], labels[val_idx]),
        "test": (images[test_idx], labels[test_idx]),
    }

    for name, (imgs, lbls) in splits.items():
        print(f"[data] {name}: {len(imgs)} samples")

    return splits


def batch_iterator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    """Infinite generator yielding (batch_images, batch_labels) tuples.

    Set shuffle=False for a single deterministic pass.
    """
    rng = np.random.default_rng(seed)
    n = len(images)
    indices = np.arange(n)

    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield images[batch_idx], labels[batch_idx]
        if not shuffle:
            break


def num_batches(n_samples: int, batch_size: int) -> int:
    """Number of batches per epoch."""
    return int(np.ceil(n_samples / batch_size))
