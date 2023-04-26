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
import shutil
import functools
from typing import Dict, List
import argparse
from pathlib import Path
from absl import logging
from flax import linen as nn
from flax.training import train_state
import flax.jax_utils as jax_utils
import jax
import jax.numpy as jnp

import ml_collections
import numpy as np
import optax
# import tensorflow_datasets as tfds
import wandb
import orbax.checkpoint

from maskgit.nets import vqgan_tokenizer, maskgit_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.model import ImageNet_class_conditional_generator_module
from maskgit.data import NumpyLoader, get_datasets
from maskgit.libml.losses import weighted_sequence_cross_entropy_loss

print(f"Total devices: {jax.device_count()}, "
      f"Devices per task: {jax.local_device_count()}")

# @jax.jit
def train_step(state: train_state.TrainState, batch, model_rng):
    """Train for a single step. """

    # https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html#example
    @functools.partial(jax.pmap, axis_name="batch")
    def train_update(state: train_state.TrainState, batch, model_rng):
        """Perform a single training step."""
        state = jax_utils.replicate(state)
        batch = jax.tree_map(lambda x: x.reshape(jax.local_device_count(), -1, *x.shape[1:]), batch)
        def loss_fn(params):
            logits, code_labels, init_mask = state.apply_fn({'params': params}, batch, model_rng)
            loss = weighted_sequence_cross_entropy_loss(
                labels=code_labels,
                logits=logits,
                weights=init_mask.astype(jnp.float32),
                label_smoothing=0.1,
            ).mean()
            return loss, (logits, code_labels)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        outs, grads = jax.pmap(grad_fn, axis_name='batch')(state.params)
        loss, (logits, code_labels) = outs
        # reduce gradients first so we save memory when it comes time to apply gradients?
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        accuracy = jax.lax.pmean(jnp.mean(jnp.argmax(logits, -1) == code_labels), axis_name='batch')
        state = jax_utils.unreplicate(state)
        state = state.apply_gradients(grads=grads)
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
        }
        return state, metrics

    return train_update(state, batch, model_rng)

@jax.jit
def predict_step(state: train_state.TrainState, batch, model_rng):
    logits, code_labels, init_mask = state.apply_fn({'params': state.params}, batch, model_rng)
    loss = weighted_sequence_cross_entropy_loss(
        labels=code_labels,
        logits=logits,
        weights=init_mask.astype(jnp.float32),
        label_smoothing=0.1,
    ).mean()
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == code_labels)
    }
    return metrics

def accumulate_metrics(batch_metrics: List[Dict[str, np.ndarray]]):
    return {
        k: jnp.mean(jnp.array([metrics[k] for metrics in batch_metrics])) \
            for k in batch_metrics[0].keys()
    }

def train_epoch(state, train_dl, model_rng, global_step=0):
    train_batch_metrics = []
    for i, batch in enumerate(train_dl):
        state, metrics = train_step(state, batch, model_rng)
        if global_step + i % 10 == 0:
            logging.info('train step %d: %s', global_step + i, metrics)
            wandb.log({
                'train/loss': metrics['loss'],
                'train/accuracy': metrics['accuracy'],
            }, step=global_step + i)
        train_batch_metrics.append(metrics)
        if i > 10:
            break # testing checkpointing.
    train_batch_metrics = accumulate_metrics(train_batch_metrics)

    return state, train_batch_metrics, global_step

def test_epoch(state, test_dl, model_rng, global_step=0):
    test_batch_metrics = []
    for batch in test_dl:
        metrics = predict_step(state, batch, model_rng)
        test_batch_metrics.append(metrics)
    test_batch_metrics = accumulate_metrics(test_batch_metrics)
    wandb.log({
        'val/loss': metrics['loss'],
        'val/accuracy': metrics['accuracy'],
    }, step=global_step)
    return test_batch_metrics

# https://flax.readthedocs.io/en/latest/guides/transfer_learning.html
from flax.core.frozen_dict import freeze

def create_train_state(rng, config, model: ImageNet_class_conditional_generator_module, pt_vars):
    """Creates initial `TrainState`."""
    # JY: base training has warmup and standard things but we're fine-tuning, hopefully this works without those complications
    x = (jnp.ones((1, 256, 256, 3), jnp.float32), jnp.ones((1,), jnp.int32))
    init_rng, call_rng = jax.random.split(rng)
    variables = model.init(init_rng, x, call_rng)
    params = variables['params']
    params = params.unfreeze()
    # params['transformer_model'] = pt_vars['params']
    params = freeze(params)
    tx = optax.adamw(config.optimizer.lr, b1=config.optimizer.beta1, b2=config.optimizer.beta2, weight_decay=config.optimizer.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
    return state

def init_ckpts(config: ml_collections.ConfigDict):
    ckpt_path = Path(config.checkpoint_dir) / wandb.run.name
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    ckpt_path.mkdir(parents=True)
    mngr = orbax.checkpoint.CheckpointManager(
        ckpt_path,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    )
    return mngr

def do_ckpt(mngr, step, state):
    mngr.save(step, state)
    breakpoint()

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

    wandb.init(
        project="maskgit_edit",
        job_type="tune",
        config=config.to_dict(),
    )
    wandb.run.name = f"{config.tag}_{wandb.run.id}"

    rng, init_rng = jax.random.split(rng)
    transformer, transformer_vars = ImageNet_class_conditional_generator_module.load_transformer_model_and_vars(config)
    model = ImageNet_class_conditional_generator_module(transformer)
    state = create_train_state(init_rng, config, model, transformer_vars)

    best_loss = 1e9
    step = 0
    mngr = init_ckpts(config)

    for epoch in range(1, config.num_epochs + 1):
        rng, model_rng = jax.random.split(rng)
        state, train_metrics, step = train_epoch(state, train_dl, model_rng, global_step=step)
        test_metrics = test_epoch(state, test_dl, model_rng)

        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            do_ckpt(mngr, state)
            logging.info(f"Saved checkpoint at step {step} val loss: {best_loss:.4f} acc: {test_metrics['loss']:.2f}")

    return state

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and evaluation script for maskgit_class_cond model")

    parser.add_argument("--batch_size", "-bs", type=int, default=2, help="Batch size for training and evaluation")
    parser.add_argument("--tag", "-t", type=str, default="default", help="Tag for wandb logging")

    return parser.parse_args()

def main():
    args = parse_arguments()

    cf = maskgit_class_cond_config.get_config()
    cf.batch_size = args.batch_size * jax.device_count()
    cf.tag = args.tag

    train_and_evaluate(cf)

if __name__ == '__main__':
    main()