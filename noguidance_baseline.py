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

import maskgit
from maskgit.utils import (
    visualize_images, read_image_from_file, restore_from_path,
    draw_image_with_bbox, Bbox,
    # draw_image_with_mask # TODO implement a highlight
)
from maskgit.inference import ImageNet_class_conditional_generator

from maskgit.notebook_utils import download_if_needed, imagenet_categories
download_if_needed()

#%%
image_size = 256
generator_256 = ImageNet_class_conditional_generator(image_size=image_size)
arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

run_mode = 'normal'  #@param ['normal', 'pmap']
# run_mode = 'pmap'  #@param ['normal', 'pmap']

p_generate_256_samples = generator_256.p_generate_samples()
p_edit_256_samples = generator_256.p_edit_samples()


#%%
category = imagenet_categories[1]
label = int(category.split(')')[0])
src_image_label = label # OK, now load from
print(src_image_label)
gt_dir = Path('./data/imagenet') / str(src_image_label)
src_dir = Path('./data/imagenet_stroke/') / str(src_image_label) / 'stroke'
mask_dir = Path('./data/imagenet_stroke/') / str(src_image_label) / 'mask'

images = list(src_dir.glob('*'))
num_examples = 20
fns = images[:num_examples]
fns = fns[-1:]
print(fns)

def get_data(fn: Path):
    return {
        'target': read_image_from_file(gt_dir / fn.name, height=image_size, width=image_size),
        'stroke': read_image_from_file(src_dir / fn.name, height=image_size, width=image_size),
        'mask': read_image_from_file(mask_dir / fn.name, height=image_size, width=image_size),
    }

import matplotlib.pyplot as plt

MODE = 'bbox'
MODE = 'mask'

SRC = 'target'
# SRC = 'stroke'

image = get_data(fns[0])[SRC]
if MODE == 'bbox':
    bbox_top_left_height_width = '64_32_128_144' # @param

    bbox = Bbox(bbox_top_left_height_width)
    draw_image_with_bbox(image, bbox)
    latent_mask, input_tokens = generator_256.create_latent_mask_and_input_tokens_for_image_editing(
        image, bbox, label)
else:
    mask = get_data(fns[0])['mask']
    latent_mask, input_tokens = generator_256.create_latent_mask_and_input_tokens_for_image_editing(
        image, bbox=None, target_label=label, mask=mask)
    fig, ax = plt.subplots()
    plt.imshow(image)
    ax.imshow(image)
    ax.axis("off")
    plt.title("input")
    plt.show()



pmap_input_tokens = generator_256.pmap_input_tokens(input_tokens)

#%%
rng, sample_rng = jax.random.split(rng)

if run_mode == 'normal':
    # starting from [2] to represent the fact that we
    # already know some tokens from the given image
    results = generator_256.generate_samples(
        input_tokens,
        sample_rng,
        start_iter=2,
        num_iterations=12
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
visualize_images(composite_images, title=f'outputs')