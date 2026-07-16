"""
CNN model definition using Flax (JAX).

Architecture:
  5 Conv2D blocks (32→64→128→256→512) with BatchNorm + ReLU + MaxPool
  → Flatten → Dense(1024) + ReLU → Dropout(0.5) → Dense(num_classes)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn


class PokemonCNN(nn.Module):
    """Convolutional neural network for Pokemon image classification.

    Args:
        num_classes: number of output classes (Pokemon species).
    """

    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # ── Block 1 ──────────────────────────────────────────────
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # ── Block 2 ──────────────────────────────────────────────
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # ── Block 3 ──────────────────────────────────────────────
        x = nn.Conv(features=128, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # ── Block 4 ──────────────────────────────────────────────
        x = nn.Conv(features=256, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # ── Block 5 ──────────────────────────────────────────────
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # ── Classifier head ──────────────────────────────────────
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=1024)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not train)(x)
        x = nn.Dense(features=self.num_classes)(x)

        return x


def initialize_model(rng, input_shape: tuple[int, ...], num_classes: int):
    """Create and initialize the PokemonCNN model.

    Args:
        rng: JAX PRNG key.
        input_shape: shape of a single image, e.g. (128, 128, 3).
        num_classes: number of Pokemon species.

    Returns:
        params: model parameters (pytree).
        batch_stats: BatchNorm statistics (pytree).
    """
    model = PokemonCNN(num_classes=num_classes)
    dummy_input = jnp.ones((1, *input_shape), dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=True)
    return variables["params"], variables.get("batch_stats", {})


def count_params(params) -> int:
    """Return total number of trainable parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
