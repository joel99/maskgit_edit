#%%
r"""
    # Structured after `MaskGIT_demo.ipynb`.
    A basic baseline which will incorporate hints by virtue of them spanning multiple patches.

    We need to
    1. Load a stroke image with its mask
    2. Provide this mask to the model as the pieces it needs to fill in
    3. Give it a number of steps to use, maxing at # of steps.

    In the model.
    4. Randomly subset the remaining tokens using the steps (model might already do this)
"""
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
import os
import itertools
from timeit import default_timer as timer
from pathlib import Path
import cv2

import matplotlib.pyplot as plt

import maskgit
from maskgit.utils import (
    visualize_images, read_image_from_file, restore_from_path,
    draw_image_with_bbox, Bbox,
    # draw_image_with_mask # TODO implement a highlight
)
from maskgit.inference import ImageNet_class_conditional_generator
from maskgit.model import ImageNet_class_conditional_generator_module

from maskgit.notebook_utils import (
    download_if_needed, imagenet_categories, load_class_to_id_map
)
download_if_needed()

wandb_id = "ytgj3lyd" # reweight 1e-5
wandb_id = "om8dz3k5" # reweight 1e-4 - curve has no difference, stepping thru notebook also shows little difference
# wandb_id = "lncehv2l" # iterate
tune_style = "iterate" if wandb_id in [
    "lncehv2l",
] else "reweight"

image_size = 256
generator_256 = ImageNet_class_conditional_generator(image_size=256, wandb_id=wandb_id, tune_style=tune_style)
generator_256.maskgit_cf.eval_batch_size = 4
arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

run_mode = 'normal'  #@param ['normal', 'pmap']
# run_mode = 'pmap'  #@param ['normal', 'pmap']

p_generate_256_samples = generator_256.p_generate_samples()
p_edit_256_samples = generator_256.p_edit_samples()

category_class_to_id = load_class_to_id_map()
#%%
category = imagenet_categories[1]
label = int(category.split(')')[0])

category_id_pth = f'n{category_class_to_id[label]}'
split = 'val'

gt_dir = Path('./data/imagenet_full') / split / category_id_pth
src_dir = Path('./data/imagenet_stroke/') / split  / category_id_pth / 'stroke'
mask_dir = Path('./data/imagenet_stroke/') / split / category_id_pth / 'mask'

images = list(src_dir.glob('*'))
num_examples = 20
fns = images[:num_examples]
fns = fns[-1:]
print(fns)

def get_data(fn: Path):
    return {
        'target': read_image_from_file(gt_dir / fn.name, height=image_size, width=image_size),
        'stroke': read_image_from_file(src_dir / fn.name, height=image_size, width=image_size),
        'mask': read_image_from_file(mask_dir / fn.name, height=image_size, width=image_size, ext='png'),
    }


MODE = 'bbox'
MODE = 'mask'

# SRC = 'target'
SRC = 'stroke'
CONTEXT_GUIDANCE = False
# CONTEXT_GUIDANCE = True

SELF_GUIDANCE_CONFIDENCE = 0.8
# SELF_GUIDANCE_STYLE = 'l2'
SELF_GUIDANCE_STYLE = "learned"
# SELF_GUIDANCE_STYLE = "iterate"

image = get_data(fns[0])[SRC]

if MODE == 'bbox':
    bbox_top_left_height_width = '64_32_128_144' # @param

    bbox = Bbox(bbox_top_left_height_width)
    draw_image_with_bbox(image, bbox)
    latent_mask, input_tokens, guidance_tokens, codebook = generator_256.create_latent_mask_and_input_tokens_for_image_editing(
        image, bbox, label)
else:
    mask = get_data(fns[0])['mask']
    latent_mask, input_tokens, guidance_tokens, codebook = generator_256.create_latent_mask_and_input_tokens_for_image_editing(
        image, bbox=None, target_label=label, mask=mask)

    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_mask = gray_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bold outline around the mask (thickness=3)
    outline_image = image.copy()
    cv2.drawContours(outline_image, contours, -1, (0, 1, 0), 2)

    fig, ax = plt.subplots()
    plt.imshow(outline_image)
    ax.imshow(outline_image)
    ax.axis("off")
    plt.title(category)
    # plt.show()

pmap_input_tokens = generator_256.pmap_input_tokens(input_tokens)

#%%
# rng, sample_rng = jax.random.split(rng)

if run_mode == 'normal':
    # starting from [2] to represent the fact that we
    # already know some tokens from the given image
    results = generator_256.generate_samples(
        input_tokens,
        sample_rng,
        start_iter=2,
        num_iterations=12,
        guidance=None if not (CONTEXT_GUIDANCE or SELF_GUIDANCE_CONFIDENCE) else guidance_tokens,
        codebook=codebook,
        context_guidance=CONTEXT_GUIDANCE,
        self_guidance_style=SELF_GUIDANCE_STYLE,
        self_guidance_lambda=SELF_GUIDANCE_CONFIDENCE,
        )

elif run_mode == 'pmap':
    sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
    results = p_edit_256_samples(pmap_input_tokens, sample_rngs)
    # flatten the pmap results
    results = results.reshape([-1, image_size, image_size, 3])

#-----------------------
# Post-process by applying a gaussian blur using the input
# and output images.
composite_images = generator_256.composite_outputs(image, latent_mask, results)
#-----------------------

visualize_images(composite_images, figsize=(12, 12), )
# visualize_images(composite_images, figsize=(12, 12), title=f'Temp: {SELF_GUIDANCE_FIDELITY}')
# visualize_images(composite_images, title=f'outputs')