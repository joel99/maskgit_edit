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
import matplotlib.pyplot as plt
import cv2

import maskgit
from maskgit.utils import (
    visualize_images_batch, read_image_from_file, restore_from_path,
    draw_image_with_bbox, Bbox,
    # draw_image_with_mask # TODO implement a highlight
)
from maskgit.inference import ImageNet_class_conditional_generator
from maskgit.model import ImageNet_class_conditional_generator_module

from maskgit.notebook_utils import (
    download_if_needed, imagenet_categories, load_class_to_id_map
)
download_if_needed()

OUTPUT_DIR = Path('./output')
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()

arbitrary_seed = 42
rng = jax.random.PRNGKey(arbitrary_seed)

category_class_to_id = load_class_to_id_map()

def load_generator(tune_style="iterate"): # or reweight, or ""
    if tune_style == "iterate":
        wandb_id = "lncehv2l"
    elif tune_style == "reweight":
        wandb_id = "om8dz3k5"
    else:
        wandb_id = ""
    generator_256 = ImageNet_class_conditional_generator(image_size=256, wandb_id=wandb_id, tune_style=tune_style)
    generator_256.maskgit_cf.eval_batch_size = 2
    return generator_256

def generate_image_with_mask_outline(image, mask):
    # Convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_mask = gray_mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bold outline around the mask (thickness=3)
    outline_image = image.copy()
    cv2.drawContours(outline_image, contours, -1, (0, 255, 0), 3)

    return outline_image

rng, sample_rng = jax.random.split(rng) # Only one source of randomness... that's fine
def generate_inference(
    image, mask, label, category,
    generator: ImageNet_class_conditional_generator,
    context_guidance=False, # true
    self_guidance_confidence=0.75,
    self_guidance_style="learned", # l2, iterate, eye, ""
    cache_outs={},
    tag="",
):
    if len(cache_outs) == 4:
        input_tokens = cache_outs['input_tokens']
        guidance_tokens = cache_outs['guidance_tokens']
        codebook = cache_outs['codebook']
        latent_mask = cache_outs['latent_mask']
    else:
        latent_mask, input_tokens, guidance_tokens, codebook = generator.create_latent_mask_and_input_tokens_for_image_editing(
            image, bbox=None, target_label=label, mask=mask)
        cache_outs['input_tokens'] = input_tokens
        cache_outs['guidance_tokens'] = guidance_tokens
        cache_outs['codebook'] = codebook
        cache_outs['latent_mask'] = latent_mask

    results = generator.generate_samples(
        input_tokens,
        sample_rng,
        start_iter=2,
        num_iterations=12,
        guidance=guidance_tokens if (context_guidance or (self_guidance_confidence and self_guidance_style != "")) else None,
        codebook=codebook,
        context_guidance=context_guidance,
        self_guidance_style=self_guidance_style,
        self_guidance_lambda=self_guidance_confidence,
        )

    composite_images = generator.composite_outputs(image, latent_mask, results)
    category_cln = category.split(',')[0].split()[1]
    if tag == "":
        tag = f"self_{self_guidance_style}_conf_{self_guidance_confidence}_ctx_{context_guidance}"
    visualize_images_batch(composite_images, output_path=OUTPUT_DIR / f'cls_{category_cln}_{tag}.png')
    return cache_outs

EVALUATION_MODES = [
    {'context_guidance': False, 'self_guidance_style': '', 'tune_style': '', 'tag': 'baseline'}, # baseline
    {'context_guidance': True, 'self_guidance_style': '', 'tune_style': '', 'tag': 'ctx_only'},
    {'context_guidance': False, 'self_guidance_style': 'eye', 'tune_style': '', 'tag': 'eye'},
    {'context_guidance': False, 'self_guidance_style': 'l2', 'tune_style': '', 'tag': 'l2'},
    {'context_guidance': False, 'self_guidance_style': 'learned', 'tune_style': 'reweight', 'tag': 'reweight'},
    {'context_guidance': False, 'self_guidance_style': 'learned', 'tune_style': 'reweight', 'tune_confidence': 0., 'tag': 'reweight_0'}, # Just demonstrate a bad one
    {'context_guidance': False, 'self_guidance_style': 'learned', 'tune_style': 'reweight', 'tune_confidence': 1., 'tag': 'reweight_1'}, # Just demonstrate a bad one
    {'context_guidance': False, 'self_guidance_style': '', 'tune_style': 'iterate', 'tag': 'iterate'},
]

def generate_one_stroke_evaluation(
    imagenet_index=1, imagenet_split='val', counterfactual=False,
    num_examples=3, image_size=256,
    evaluation_modes=EVALUATION_MODES
):

    category = imagenet_categories[imagenet_index]
    label = int(category.split(')')[0])
    category_id_pth = f'n{category_class_to_id[label]}'
    gt_dir = Path('./data/imagenet_full') / imagenet_split / category_id_pth
    counterfactual_suffix = '_counterfactual' if counterfactual else ''
    src_dir = Path('./data/imagenet_stroke/') / imagenet_split  / category_id_pth / f'stroke{counterfactual_suffix}'
    mask_dir = Path('./data/imagenet_stroke/') / imagenet_split / category_id_pth / f'mask{counterfactual_suffix}'

    images = list(src_dir.glob('*'))
    fns = images[:num_examples]

    def get_data(fn: Path):
        return {
            'target': read_image_from_file(gt_dir / fn.name, height=image_size, width=image_size),
            'stroke': read_image_from_file(src_dir / fn.name, height=image_size, width=image_size),
            'mask': read_image_from_file(mask_dir / fn.name, height=image_size, width=image_size, ext='png'),
        }

    # Not at all efficient, c'est la vie

    for i, fn in enumerate(fns):
        image = get_data(fn)['stroke']
        mask = get_data(fn)['mask']
        outline_image = generate_image_with_mask_outline(image, mask)

        fig, ax = plt.subplots()
        plt.imshow(outline_image)
        ax.imshow(outline_image)
        ax.axis("off")
        plt.title(category)
        output_path = os.path.join(OUTPUT_DIR, f'cls-{label}_{i}.png')
        plt.savefig(output_path)
        plt.close(fig)
        cache = {}
        for mode in evaluation_modes:
            print(mode)
            cache = generate_inference(
                image, mask, label, category,
                generator=load_generator(tune_style=mode['tune_style']),
                context_guidance=mode['context_guidance'],
                self_guidance_style=mode['self_guidance_style'],
                self_guidance_confidence=mode.get('tune_confidence', 0.5),
                tag=mode['tag'],
                cache_outs=cache
            )
generate_one_stroke_evaluation(imagenet_index=1, counterfactual=False)