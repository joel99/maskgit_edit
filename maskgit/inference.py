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
import os.path as osp
import io
import flax
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
import orbax.checkpoint
# import tensorflow.compat.v1 as tf

from maskgit.nets import vqgan_tokenizer, maskgit_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.libml import parallel_decode
from maskgit.utils import restore_from_path

#TODO: this can be usedforediting aswell; justneedto pass in a different start_iter
#TODO: perhaps move rng out of  this class?
class ImageNet_class_conditional_generator():
    def checkpoint_canonical_path(maskgit_or_tokenizer, image_size):
        return f"./checkpoints/{maskgit_or_tokenizer}_imagenet{image_size}_checkpoint"

    def __init__(self, image_size=256, wandb_id="", tune_style="iterate"):
        maskgit_cf = maskgit_class_cond_config.get_config()
        maskgit_cf.image_size = int(image_size)
        maskgit_cf.eval_batch_size = 8

        # Define tokenizer
        self.tokenizer_model = vqgan_tokenizer.VQVAE(config=maskgit_cf, dtype=jnp.float32, train=False)

        # Define transformer
        self.transformer_latent_size = maskgit_cf.image_size // maskgit_cf.transformer.patch_size
        self.transformer_codebook_size = maskgit_cf.vqvae.codebook_size + maskgit_cf.num_class + 1
        self.transformer_block_size = self.transformer_latent_size ** 2 + 1
        self.transformer_model = maskgit_transformer.Transformer(
            vocab_size=self.transformer_codebook_size,
            hidden_size=maskgit_cf.transformer.num_embeds,
            num_hidden_layers=maskgit_cf.transformer.num_layers,
            num_attention_heads=maskgit_cf.transformer.num_heads,
            intermediate_size=maskgit_cf.transformer.intermediate_size,
            hidden_dropout_prob=maskgit_cf.transformer.dropout_rate,
            attention_probs_dropout_prob=maskgit_cf.transformer.dropout_rate,
            max_position_embeddings=self.transformer_block_size)

        self.maskgit_cf = maskgit_cf
        if tune_style == "reweight":
            self.reweight_kernel = jnp.ones((self.maskgit_cf.vqvae.codebook_size, self.maskgit_cf.vqvae.codebook_size), dtype=jnp.float32)
        else:
            self.reweight_kernel = None

        self._load_checkpoints(wandb_id)

    def _load_checkpoints(self, wandb_id=""):
        image_size = self.maskgit_cf.image_size

        self.tokenizer_variables = restore_from_path(
            ImageNet_class_conditional_generator.checkpoint_canonical_path("tokenizer", image_size))
        if wandb_id:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            wandb_dir = f"./checkpoints/default_{wandb_id}/"
            # get latest subdir by time, which is lowest loss
            latest_subdir = sorted(os.listdir(wandb_dir), key=lambda x: os.path.getmtime(os.path.join(wandb_dir, x)))[-1]
            raw_restored = orbax_checkpointer.restore(osp.join(wandb_dir, latest_subdir, 'default'))['params']
            # breakpoint()
            self.transformer_variables = {'params': raw_restored["transformer_model"]}
            if self.reweight_kernel is not None:
                self.reweight_kernel = jnp.array(raw_restored["reweight_kernel"])
                # JY: Note that the actual learned kernel is scarcely different than the initial kernel by manual inspection of weights (i.e. 8.44 -> 8.45)
        else:
            self.transformer_variables = restore_from_path(
                ImageNet_class_conditional_generator.checkpoint_canonical_path("maskgit", image_size))

    @staticmethod
    def get_guidance_energy(
        self_guidance_style, codebook # N x D
    ): # return N x N
        # technically None should also provide clean implementation
        if self_guidance_style == "l2":
            difference = codebook[:, None, :] - codebook[None, :, :] # N x N x D
            return -jnp.mean(difference ** 2, axis=-1) # N x N
        elif self_guidance_style == "eye":
            return jnp.eye(codebook.shape[0])
        elif self_guidance_style == "knn":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def generate_samples(
        self,
        input_tokens,
        rng,
        start_iter=0,
        num_iterations=16,
        guidance=None,
        codebook=None,
        context_guidance=False,
        self_guidance_lambda=1.0,
        self_guidance_style="l2", # [iterate, eye, l2, learned]
    ):
        breakpoint()
        if self_guidance_style == "iterate":
            def tokens_to_logits(seq, guide):
                logits = self.transformer_model.apply(self.transformer_variables, seq, guidance_ids=guide, deterministic=True) # TODO use guide
                logits = logits[..., :self.maskgit_cf.vqvae.codebook_size]
                return logits
            output_tokens = parallel_decode.decode_self_guidance_iterate(
                input_tokens,
                guidance,
                rng,
                tokens_to_logits,
                num_iter=num_iterations,
                choice_temperature=self.maskgit_cf.sample_choice_temperature,
                mask_token_id=self.maskgit_cf.transformer.mask_token_id,
                start_iter=start_iter,
                self_guidance=None,
                confidence=self_guidance_lambda,
            )
        else:
            def tokens_to_logits(seq):
                logits = self.transformer_model.apply(self.transformer_variables, seq, deterministic=True)
                logits = logits[..., :self.maskgit_cf.vqvae.codebook_size]
                return logits
            assert not (context_guidance and self_guidance_lambda), "Guidance is mutually exclusive in current implementation."
            if guidance is not None:
                if context_guidance:
                    output_tokens = parallel_decode.decode_context_guidance(
                        input_tokens,
                        guidance,
                        rng,
                        tokens_to_logits,
                        codebook_size=self.maskgit_cf.vqvae.codebook_size,
                        num_iter=num_iterations,
                        choice_temperature=self.maskgit_cf.sample_choice_temperature,
                        mask_token_id=self.maskgit_cf.transformer.mask_token_id,
                        start_iter=start_iter,
                    )
                else:
                    if self_guidance_style == "learned":
                        assert self.reweight_kernel is not None
                        self_guidance = self.reweight_kernel
                    else:
                        self_guidance = ImageNet_class_conditional_generator.get_guidance_energy(self_guidance_style, codebook)
                    output_tokens = parallel_decode.decode_self_guidance(
                        input_tokens,
                        guidance,
                        rng,
                        tokens_to_logits,
                        num_iter=num_iterations,
                        choice_temperature=self.maskgit_cf.sample_choice_temperature,
                        mask_token_id=self.maskgit_cf.transformer.mask_token_id,
                        start_iter=start_iter,
                        self_guidance=self_guidance,
                        confidence=self_guidance_lambda,
                    )
            else:
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

    def _create_latent_mask_from_bbox(self, bbox):
        latent_mask = np.zeros((self.maskgit_cf.eval_batch_size,
                                self.transformer_latent_size,
                                self.transformer_latent_size))
        latent_t = max(0, bbox.top // self.maskgit_cf.transformer.patch_size - 1)
        latent_b = min(self.transformer_latent_size,
                    bbox.height // self.maskgit_cf.transformer.patch_size + bbox.top // self.maskgit_cf.transformer.patch_size + 1)
        latent_l = max(0, bbox.left // self.maskgit_cf.transformer.patch_size - 1)
        latent_r = min(self.transformer_latent_size,
                    bbox.left // self.maskgit_cf.transformer.patch_size + bbox.width // self.maskgit_cf.transformer.patch_size + 1)

        latent_mask[:, latent_t:latent_b, latent_l:latent_r] = 1

        return latent_mask

    def _create_latent_mask_from_mask(self, mask):
        if len(mask.shape) == 3:
            mask = mask.any(axis=-1)
        # unsqueeze for eval batch size
        mask = np.repeat(mask[None], self.maskgit_cf.eval_batch_size, axis=0)

        latent_mask = np.zeros((self.maskgit_cf.eval_batch_size,
                                self.transformer_latent_size,
                                self.transformer_latent_size))

        # Downsample the input mask
        downsampled_mask = mask.reshape(
            self.maskgit_cf.eval_batch_size,
            self.transformer_latent_size, self.maskgit_cf.transformer.patch_size,
            self.transformer_latent_size, self.maskgit_cf.transformer.patch_size
        ).any(axis=(2, 4))

        # Check if any pixel in the PATCH_SIZE x PATCH_SIZE patch is masked and set the corresponding entry in the latent_mask to 1
        latent_mask[downsampled_mask] = 1

        return latent_mask

    def create_latent_mask_and_input_tokens_for_image_editing(
        self,
        image,
        bbox=None,
        target_label=0,
        mask=None, # mask matching image size indicating edited region
    ):
        assert bbox is not None or mask is not None, "Either bbox or mask must be provided"
        imgs = self._create_input_batch(image)

        # Encode the images into image tokens
        image_tokens = self.tokenizer_model.apply(
              self.tokenizer_variables,
              {"image": imgs},
              method=self.tokenizer_model.encode_to_indices,
              mutable=False)

        codebook = self.tokenizer_model.apply(
            self.tokenizer_variables,
            method=self.tokenizer_model.get_codebook_funct,
            mutable=False
        )

        # Create the masked tokens
        if mask is not None:
            latent_mask = self._create_latent_mask_from_mask(mask)
        else:
            latent_mask = self._create_latent_mask_from_bbox(bbox)

        masked_tokens = (1-latent_mask) * image_tokens + self.maskgit_cf.transformer.mask_token_id * latent_mask
        masked_tokens = np.reshape(masked_tokens, [self.maskgit_cf.eval_batch_size, -1])

        # Create input tokens based on the category label
        label_tokens = target_label * jnp.ones([self.maskgit_cf.eval_batch_size, 1])
        # Shift the label tokens by codebook_size
        label_tokens = label_tokens + self.maskgit_cf.vqvae.codebook_size
        # Concatenate the two as input_tokens
        input_tokens = jnp.concatenate([label_tokens, masked_tokens], axis=-1)

        # The usage of guidance tokens implies the input image is already edited.
        guidance_tokens = np.reshape(image_tokens, [self.maskgit_cf.eval_batch_size, -1])
        guidance_tokens = jnp.concatenate([label_tokens, guidance_tokens], axis=-1) # * Technically just saying class token as guidance for class token - useful for matching shape in context_guidance but probably not important.
        return (latent_mask, input_tokens.astype(jnp.int32), guidance_tokens.astype(jnp.int32), codebook)


    def composite_outputs(self, input, latent_mask, outputs):
        imgs = self._create_input_batch(input)
        composit_mask = Image.fromarray(np.uint8(latent_mask[0] * 255.))
        composit_mask = composit_mask.resize((self.maskgit_cf.image_size, self.maskgit_cf.image_size))
        composit_mask = composit_mask.filter(ImageFilter.GaussianBlur(radius=self.maskgit_cf.image_size//16-1))
        composit_mask = np.float32(composit_mask)[:, :, np.newaxis] / 255.
        return outputs * composit_mask + (1-composit_mask) * imgs
