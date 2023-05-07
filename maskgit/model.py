# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import flax
from flax import linen as nn
import functools
import jax
import jax.numpy as jnp
from jax import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import ImageFilter, Image
import requests
# import tensorflow.compat.v1 as tf

from maskgit.nets import vqgan_tokenizer, maskgit_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.libml import parallel_decode
from maskgit.utils import restore_from_path
from maskgit.inference import ImageNet_class_conditional_generator

#TODO: compress by subclassing the og generator
class ImageNet_class_conditional_generator_module(nn.Module):
    transformer_model: maskgit_transformer.Transformer

    def checkpoint_canonical_path(maskgit_or_tokenizer, image_size):
        return f"./checkpoints/{maskgit_or_tokenizer}_imagenet{image_size}_checkpoint"

    def setup(self, image_size=256):
        maskgit_cf = maskgit_class_cond_config.get_config()
        maskgit_cf.image_size = int(image_size)
        maskgit_cf.eval_batch_size = 8

        # Define tokenizer
        self.tokenizer_model = vqgan_tokenizer.VQVAE(config=maskgit_cf, dtype=jnp.float32, train=False)

        # Define transformer
        self.transformer_latent_size = maskgit_cf.image_size // maskgit_cf.transformer.patch_size
        self.transformer_codebook_size = maskgit_cf.vqvae.codebook_size + maskgit_cf.num_class + 1
        self.transformer_block_size = self.transformer_latent_size ** 2 + 1

        self.maskgit_cf = maskgit_cf

        self._load_checkpoints()

    def _load_checkpoints(self):
        image_size = self.maskgit_cf.image_size

        # self.transformer_variables = restore_from_path(
            # ImageNet_class_conditional_generator.checkpoint_canonical_path("maskgit", image_size))
        self.tokenizer_variables = restore_from_path(
            ImageNet_class_conditional_generator.checkpoint_canonical_path("tokenizer", image_size))

    @classmethod
    def load_transformer_model_and_vars(cls, maskgit_cf):
        transformer_latent_size = maskgit_cf.image_size // maskgit_cf.transformer.patch_size
        transformer_codebook_size = maskgit_cf.vqvae.codebook_size + maskgit_cf.num_class + 1
        transformer_block_size = transformer_latent_size ** 2 + 1
        transformer_model = maskgit_transformer.Transformer(
            vocab_size=transformer_codebook_size,
            hidden_size=maskgit_cf.transformer.num_embeds,
            num_hidden_layers=maskgit_cf.transformer.num_layers,
            num_attention_heads=maskgit_cf.transformer.num_heads,
            intermediate_size=maskgit_cf.transformer.intermediate_size,
            hidden_dropout_prob=maskgit_cf.transformer.dropout_rate,
            attention_probs_dropout_prob=maskgit_cf.transformer.dropout_rate,
            max_position_embeddings=transformer_block_size)
        return transformer_model, restore_from_path(
            ImageNet_class_conditional_generator.checkpoint_canonical_path("maskgit", maskgit_cf.image_size))

    def generate_samples(self, input_tokens, rng, start_iter=0, num_iterations=16):
        def tokens_to_logits(seq):
            logits = self.transformer_model.apply(self.transformer_variables, seq, deterministic=True)
            logits = logits[..., :self.maskgit_cf.vqvae.codebook_size]
            return logits

        output_tokens = parallel_decode.decode(
            input_tokens,
            rng,
            tokens_to_logits,
            num_iter=num_iterations,
            choice_temperature=self.maskgit_cf.sample_choice_temperature,
            mask_token_id=self.maskgit_cf.transformer.mask_token_id,
            start_iter=start_iter,
        )

        output_tokens = jnp.reshape(output_tokens[:, -1, 1:], [-1, self.transformer_latent_size, self.transformer_latent_size])
        gen_images = self.tokenizer_model.apply(
            self.tokenizer_variables,
            output_tokens,
            method=self.tokenizer_model.decode_from_indices,
            mutable=False)

        return gen_images

    def create_input_tokens_normal(self, label):
        label_tokens = label * jnp.ones([self.maskgit_cf.eval_batch_size, 1])
        # Shift the label by codebook_size
        label_tokens = label_tokens + self.maskgit_cf.vqvae.codebook_size
        # Create blank masked tokens
        blank_tokens = jnp.ones([self.maskgit_cf.eval_batch_size, self.transformer_block_size-1])
        masked_tokens = self.maskgit_cf.transformer.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        input_tokens = jnp.concatenate([label_tokens, masked_tokens], axis=-1)
        return input_tokens.astype(jnp.int32)

    def p_generate_samples(self, start_iter=0, num_iterations=16):
        """For TPUs/GPUs with lots of memory, using pmap provides a substantial speedup, but
        requires a slightly different API call and a different input shape.
        """
        return jax.pmap(functools.partial(self.generate_samples, start_iter=start_iter, num_iterations=num_iterations), axis_name="batch")

    def p_edit_samples(self, start_iter=2, num_iterations=12):
        """For TPUs/GPUs with lots of memory, using pmap provides a substantial speedup, but
        requires a slightly different API call and a different input shape.
        """
        return jax.pmap(functools.partial(self.generate_samples, start_iter=start_iter, num_iterations=num_iterations), axis_name="batch")

    def pmap_input_tokens(self, input_tokens):
        device_count = jax.local_device_count()
        input_tokens = input_tokens.reshape(
            [device_count, self.maskgit_cf.eval_batch_size // device_count, -1])
        return jax.device_put(input_tokens)

    def rng_seed(self):
        return self.maskgit_cf.seed

    def eval_batch_size(self):
        return self.maskgit_cf.eval_batch_size

    def _create_input_batch(self, image):
        return np.repeat(image[None], self.maskgit_cf.eval_batch_size, axis=0).astype(np.float32)

    def composite_outputs(self, input, latent_mask, outputs):
        imgs = self._create_input_batch(input)
        composit_mask = Image.fromarray(np.uint8(latent_mask[0] * 255.))
        composit_mask = composit_mask.resize((self.maskgit_cf.image_size, self.maskgit_cf.image_size))
        composit_mask = composit_mask.filter(ImageFilter.GaussianBlur(radius=self.maskgit_cf.image_size//16-1))
        composit_mask = np.float32(composit_mask)[:, :, np.newaxis] / 255.
        return outputs * composit_mask + (1-composit_mask) * imgs

    def __call__(self, batch, rng): # or call?
        image, target_label = batch # dtypes float and int
        # This makes no sense to me, but if pmap isn't killing the device dimension, we have to flatten it back
        if len(image.shape) == 5:
            image = np.reshape(image, [image.shape[0] * image.shape[1], *image.shape[2:]])
            target_label = np.reshape(target_label, [target_label.shape[0] * target_label.shape[1]])
        target_label = target_label[:, None]
        image_tokens = self.tokenizer_model.apply(
            self.tokenizer_variables,
            {"image": image},
            method=self.tokenizer_model.encode_to_indices,
            mutable=False)

        # TODO multi-step MVTM and assert known tokens (meh, can just do that at inference)
        # Asserting known tokens needs to be done if we want to work with stroke skyline.

        image_tokens = np.reshape(image_tokens, [image_tokens.shape[0], -1]).astype("int32")
        initial_mask_ratio = random.uniform(rng, shape=(1,)) # whatever, broadcast throws if we try to get a random per example
        latent_mask = random.bernoulli(
            rng,
            p=initial_mask_ratio,
            shape=image_tokens.shape
        )
        masked_tokens = (1-latent_mask) * image_tokens + latent_mask * self.maskgit_cf.transformer.mask_token_id

        # Shift the label tokens by codebook_size
        image_cls_tokens = target_label + self.maskgit_cf.vqvae.codebook_size
        # Concatenate the two as input_tokens
        input_tokens = jnp.concatenate([image_cls_tokens, masked_tokens], axis=-1)

        # if self.transformer_model.is_mutable_collection("params"):
        #     # TODO: low pri - this is redundantly called everytime. How can I just call it once for init? (We need to do a proper init despite pretraining bc model.init must be called before we can transfer in weights)
        #     init_vars = self.transformer_model(input_tokens) # https://flax.readthedocs.io/en/latest/api_reference/_autosummary/flax.linen.while_loop.html - consider initializing before
        output_logits = parallel_decode.decode_logit_flax_scan(
            input_tokens,
            rng,
            self.transformer_model,
            codebook_size=self.maskgit_cf.vqvae.codebook_size, # this is the codebook of the logits we care about (transformer also predicts logits for imagenet classes)
            num_iter=2,
            choice_temperature=self.maskgit_cf.sample_choice_temperature,
            mask_token_id=self.maskgit_cf.transformer.mask_token_id,
            start_iter=0,
        )
        output_logits = output_logits[:, 1:] # since class label is first token
        return output_logits, image_tokens, latent_mask # initial mask

    # Matching