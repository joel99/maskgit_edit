from pathlib import Path
import cv2
import numpy as np
from skimage import color
from skimage.segmentation import slic
from skimage.util import img_as_float
import random
import os
import argparse

def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def reduce_colors(image, n_colors):
    image_float = img_as_float(image)
    segments = slic(image_float, n_segments=n_colors, sigma=1, start_label=1)
    segmented_image = color.label2rgb(segments, image_float, kind='avg')
    return (segmented_image * 255).astype(np.uint8), segments

def overlay_stroke(original_image, reduced_image, segments):
    unique_segments = np.unique(segments)
    selected_segment = random.choice(unique_segments)

    mask = np.where(segments == selected_segment, True, False)
    overlay_image = np.where(mask[..., np.newaxis], reduced_image, original_image)

    return overlay_image, mask

def human_stroke_simulation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filtered_image = median_filter(image, kernel_size=23)
    reduced_color_image, segments = reduce_colors(filtered_image, n_colors=12)

    overlay_image, mask = overlay_stroke(image, reduced_color_image, segments)

    return overlay_image, mask

def process_images(input_dir, output_dir, random_seed):
    random.seed(random_seed)

    for subdir, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_image_path = os.path.join(subdir, filename)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = Path(output_dir) / relative_path

                if not (output_subdir / 'stroke').exists():
                    (output_subdir / 'stroke').mkdir(parents=True)
                if not (output_subdir / 'mask').exists():
                    (output_subdir / 'mask').mkdir(parents=True)
                output_image_path = os.path.join(output_subdir, 'stroke', filename)
                if os.path.exists(output_image_path):
                    continue
                output_mask_path = os.path.join(output_subdir, 'mask', filename)

                processed_image, mask = human_stroke_simulation(input_image_path)

                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_image_path, processed_image)

                cv2.imwrite(output_mask_path, mask.astype(np.uint8) * 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-stroke-simulation algorithm")
    parser.add_argument("-i", "--input-dir", type=str, default="./data/imagenet", help="Input directory containing images")
    parser.add_argument("-o", "--output-dir", type=str, default="./data/imagenet_stroke", help="Output directory for processed images and masks")
    parser.add_argument("-r", "--random-seed", type=int, default=42, help="Random seed for stroke generation")

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.random_seed)
