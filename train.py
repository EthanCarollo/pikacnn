"""
Training script for Pokemon Generation One CNN.

Downloads the dataset, trains the model, and saves the best checkpoint.
All hyperparameters are defined in the CONFIG dict at the top of the file.
"""

import os
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
from tqdm import tqdm

from data_loader import (
    download_dataset,
    load_dataset,
    split_dataset,
    batch_iterator,
    num_batches,
)
from model import PokemonCNN, initialize_model, count_params

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Data
    "image_size": (128, 128),  # (H, W)
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    # Training
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    # Early stopping
    "early_stop_patience": 10,
    # Reproducibility
    "seed": 42,
    # Output
    "checkpoint_dir": "checkpoints",
}

# ═══════════════════════════════════════════════════════════════════════════════
# JIT-compiled training & evaluation steps
# ═══════════════════════════════════════════════════════════════════════════════


@functools.partial(jax.jit, static_argnames=("model",))
def train_step(state, batch, model: PokemonCNN):
    """Single training step: forward, loss, backward, update."""

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, updates = model.apply(
            variables, batch["images"], train=True, mutable=["batch_stats"]
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["labels"]
        ).mean()
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])

    acc = (logits.argmax(axis=-1) == batch["labels"]).mean()
    return state, loss, acc


@functools.partial(jax.jit, static_argnames=("model",))
def eval_step(state, batch, model: PokemonCNN):
    """Single evaluation step (no gradients)."""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = model.apply(variables, batch["images"], train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, batch["labels"]
    ).mean()
    acc = (logits.argmax(axis=-1) == batch["labels"]).mean()
    return loss, acc


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TrainState(train_state.TrainState):
    """Extended TrainState that also carries batch_stats for BatchNorm."""
    batch_stats: dict


def create_train_state(rng, model, input_shape, config):
    """Initialise parameters, optimizer and TrainState."""
    params, batch_stats = initialize_model(rng, input_shape, model.num_classes)

    total_steps = config["epochs"] * config["steps_per_epoch"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=config["warmup_epochs"] * config["steps_per_epoch"],
        decay_steps=total_steps,
        end_value=config["learning_rate"] * 0.01,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule, weight_decay=config["weight_decay"]
        ),
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )


def run_epoch(state, model, images, labels, batch_size, config, training: bool):
    """Run one epoch — training or evaluation. Returns (state, avg_loss, avg_acc)."""
    it = batch_iterator(images, labels, batch_size, shuffle=training, seed=config["seed"])
    n_batches = num_batches(len(images), batch_size)

    total_loss = 0.0
    total_acc = 0.0

    for _ in range(n_batches):
        x_batch, y_batch = next(it)
        batch = {"images": jnp.array(x_batch), "labels": jnp.array(y_batch)}

        if training:
            state, loss, acc = train_step(state, batch, model)
        else:
            loss, acc = eval_step(state, batch, model)

        total_loss += float(loss)
        total_acc += float(acc)

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    return state, avg_loss, avg_acc


def save_checkpoint(state, config):
    """Save model parameters as an Orbax checkpoint."""
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(
        os.path.abspath(config["checkpoint_dir"]),
        args=ocp.args.StandardSave(state.params),
    )
    print(f"[save] checkpoint saved to {config['checkpoint_dir']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    config = CONFIG.copy()
    print("[train] JAX devices:", jax.devices())

    # ── Seed ───────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(config["seed"])

    # ── Data ───────────────────────────────────────────────────────
    data_dir = download_dataset()
    images, labels, class_names = load_dataset(data_dir, config["image_size"])
    config["num_classes"] = len(class_names)
    config["steps_per_epoch"] = num_batches(
        int(len(images) * (1 - config["val_ratio"] - config["test_ratio"])),
        config["batch_size"],
    )

    splits = split_dataset(
        images, labels,
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
    )

    # ── Model ──────────────────────────────────────────────────────
    model = PokemonCNN(num_classes=config["num_classes"])
    input_shape = (*config["image_size"], 3)
    state = create_train_state(rng, model, input_shape, config)
    print(f"[train] model params: {count_params(state.params):,}")

    # ── Training loop ──────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    train_images, train_labels = splits["train"]
    val_images, val_labels = splits["val"]

    for epoch in range(1, config["epochs"] + 1):
        t_start = time.perf_counter()

        # Training
        state, train_loss, train_acc = run_epoch(
            state, model, train_images, train_labels,
            config["batch_size"], config, training=True,
        )

        # Validation
        state, val_loss, val_acc = run_epoch(
            state, model, val_images, val_labels,
            config["batch_size"], config, training=False,
        )

        elapsed = time.perf_counter() - t_start
        lr = float(state.opt_state[-1].hyperparams["learning_rate"])

        # Log
        print(
            f"[epoch {epoch:3d}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={lr:.2e}  time={elapsed:.1f}s"
        )

        # Checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(state, config)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config["early_stop_patience"]:
            print(f"[train] early stopping at epoch {epoch} (best val_acc={best_val_acc:.3f})")
            break

    # ── Final test evaluation ──────────────────────────────────────
    print("\n[train] evaluating on test set ...")
    test_images, test_labels = splits["test"]
    _, test_loss, test_acc = run_epoch(
        state, model, test_images, test_labels,
        config["batch_size"], config, training=False,
    )
    print(f"[test]  loss={test_loss:.4f}  accuracy={test_acc:.3f}")
    print("[train] done.")


if __name__ == "__main__":
    main()
