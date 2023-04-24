#%%
r"""
    .py copy of the demo
"""
#%%
import subprocess
import numpy as np
import jax
import jax.numpy as jnp
import os
import itertools
from timeit import default_timer as timer

# export cuda visible devices to 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import maskgit
from maskgit.utils import visualize_images, read_image_from_url, restore_from_path, draw_image_with_bbox, Bbox
from maskgit.inference import ImageNet_class_conditional_generator
from maskgit.notebook_utils import download_if_needed, imagenet_categories
download_if_needed()

#%%
generator_256 = ImageNet_class_conditional_generator(image_size=256)
# generator_512 = ImageNet_class_conditional_generator(image_size=512)
arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

run_mode = 'normal'  #@param ['normal', 'pmap']
run_mode = 'pmap'  #@param ['normal', 'pmap']

p_generate_256_samples = generator_256.p_generate_samples()
p_edit_256_samples = generator_256.p_edit_samples()

#%%
# Class conditional image synthesis
category = imagenet_categories[0]
label = int(category.split(')')[0])

# prep the input tokens based on the chosen label
input_tokens = generator_256.create_input_tokens_normal(label)
pmap_input_tokens = generator_256.pmap_input_tokens(input_tokens)

#%%
# we default to 256 here which is faster
image_size = 256

# NOTE that in both run modes, subsequent re-runs tend to be much faster
# than the initial run.

rng, sample_rng = jax.random.split(rng)
start_timer = timer()

# In "normal" mode, a batch of 8 images takes a V80
# ~25 seconds in 256x256, and ~75 seconds in 512x512.
if run_mode == 'normal':
    results = generator_256.generate_samples(input_tokens, sample_rng)
elif run_mode == 'pmap':
    sample_rngs = jax.random.split(sample_rng, jax.local_device_count())
    results = p_generate_256_samples(pmap_input_tokens, sample_rngs)

    # flatten the pmap results
    results = results.reshape([-1, image_size, image_size, 3])

end_timer = timer()
print(f"generated {generator_256.eval_batch_size()} images in {end_timer - start_timer} seconds")

# Visualize
visualize_images(results, title=f'results')

#%%
# Class conditional image editing
category = imagenet_categories[1]
label = int(category.split(')')[0])

#%%
# we switch to 512 here for demo purposes
image_size = 256

# Feel free to change the input below to your favorite example!
bbox_top_left_height_width = '64_32_128_144' # @param
# bbox_top_left_height_width = '128_64_256_288' # @param
img_url = 'https://storage.googleapis.com/maskgit-public/imgs/class_cond_input_1.png' # @param

bbox = Bbox(bbox_top_left_height_width)

# Load the input image, and visualize it with our bounding box
image = read_image_from_url(
    img_url,
    height=image_size,
    width=image_size)

draw_image_with_bbox(image, bbox)

latent_mask, input_tokens = generator_256.create_latent_mask_and_input_tokens_for_image_editing(
    image, bbox, label)

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