r"""
    Bootstrapping from https://github.com/google/flax/blob/main/examples/mnist/train.py
    A Flax tuning script that mimics the MaskGIT MVTM training process, with the following changes:
    1. N-step decoding DURING training (vs 1-step)
    2. Add input encoding for all tokens, including masked ones.
        - TODO experiment whether the input content for masked tokens should be directly added or provided separately.
    3. Scale mask embedding according to confidence.
"""

# We keep identically the recommended training settings from the paper, but have to reproduce the training loop since they didn't release it yet.


r"""
    Tasklog
    # 0. dataloader
    # loads images, masks, and strokes. For starters we can play with just the imgs and start from step 0.
    # 1. VQ-encoding to create inputs and outputs
    #   - output targets should be on source image
    #   - we initialize at step 2, just like inference, and lock in fully unmasked
    # 2. Identify label smoothing and/or their hps
    # 3. ? train?
"""

# See issue #620.
# pytype: disable=wrong-keyword-args
import os
from typing import Dict, List
import argparse
from pathlib import Path
from absl import logging
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
# import tensorflow_datasets as tfds
import wandb

from maskgit.nets import vqgan_tokenizer, maskgit_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.inference import ImageNet_class_conditional_generator
from maskgit.data import NumpyLoader, get_datasets

@jax.jit
def train_step(state: train_state.TrainState, batch, model_rng):
    """Train for a single step. """
    breakpoint()
    def loss_fn(params):
        logits, code_labels = state.apply_fn({'params': params}, batch, model_rng)
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=code_labels).mean()
        return loss, logits, code_labels
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits, code_labels), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == jnp.max(code_labels, -1))
    }
    return state, metrics

@jax.jit
def predict_step(state: train_state.TrainState, batch, model_rng):
    logits, code_labels = state.apply_fn({'params': state.params}, batch, model_rng)
    loss = optax.softmax_cross_entropy(
        logits=logits, labels=code_labels).mean()
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == jnp.max(code_labels, -1))
    }
    return metrics

def accumulate_metrics(batch_metrics: List[Dict[str, np.ndarray]]):
    return {
        k: jnp.mean([metrics[k] for metrics in batch_metrics]) \
            for k in batch_metrics[0].keys()
    }

def train_epoch(state, train_dl, model_rng):
    train_batch_metrics = []
    for batch in train_dl:
        state, metrics = train_step(state, batch, model_rng)
        train_batch_metrics.append(metrics)
    train_batch_metrics = accumulate_metrics(train_batch_metrics)

    return state, train_batch_metrics

def test_epoch(state, test_dl, model_rng):
    test_batch_metrics = []
    for batch in test_dl:
        metrics = predict_step(state, batch, model_rng)
        test_batch_metrics.append(metrics)
    test_batch_metrics = accumulate_metrics(test_batch_metrics)
    return test_batch_metrics

def create_train_state(rng, config, model: ImageNet_class_conditional_generator):
    """Creates initial `TrainState`."""
    # JY: base training has warmup and standard things but we're fine-tuning, hopefully this works without those complications
    tx = optax.adamw(config.optimizer.lr, b1=config.optimizer.beta1, b2=config.optimizer.beta2, weight_decay=config.optimizer.weight_decay)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=model.transformer_variables['params'], tx=tx)

def train_and_evaluate(config: ml_collections.ConfigDict) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
        config: Hyperparameter configuration for training and evaluation.

    Returns:
        The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    train_dl = NumpyLoader(
        train_ds, batch_size=config.batch_size, num_workers=8,
        shuffle=True
    )
    test_dl = NumpyLoader(
        test_ds, batch_size=config.batch_size, num_workers=1
    )

    rng = jax.random.PRNGKey(config.seed)

    # wandb.init(
    #     project="maskgit_edit",
    #     job_type="tune",
    #     config=config.to_dict(),
    # )

    rng, init_rng = jax.random.split(rng)
    model = ImageNet_class_conditional_generator()
    state = create_train_state(init_rng, config, model)

    for epoch in range(1, config.num_epochs + 1):
        rng, model_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_dl, model_rng)
        test_loss, test_accuracy = test_epoch(state, test_dl, model_rng)

        logging.info(
            'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f'
            % (epoch, train_loss, train_accuracy * 100, test_loss,
            test_accuracy * 100)
        )
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": test_loss,
            "Validation Accuracy": test_accuracy
        }, step=epoch)

    return state

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and evaluation script for maskgit_class_cond model")

    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size for training and evaluation")

    return parser.parse_args()

def main():
    args = parse_arguments()

    cf = maskgit_class_cond_config.get_config()
    cf.batch_size = args.batch_size

    train_and_evaluate(cf)

if __name__ == '__main__':
    main()