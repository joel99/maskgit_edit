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
import functools
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import ImageFilter, Image
import requests
import tensorflow.compat.v1 as tf

def visualize_images_batch(images, title='', figsize_unit=6, output_path="output.png"):
  batch_size, height, width, c = images.shape
  n_cols = 1
  n_rows = batch_size
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_unit, n_rows * figsize_unit))
  axes = axes.flatten()

  for idx, (image, ax) in enumerate(zip(images, axes)):
      image = np.clip(image, 0, 1)
      ax.imshow(image)
      ax.axis("off")
  # save to output_path
  plt.suptitle(title)
  plt.tight_layout()
  plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
  # clear
  plt.clf()
  # plt.close()


def visualize_images(images, title='', figsize=(30, 6)):
  batch_size, height, width, c = images.shape
  n_rows = int(math.sqrt(batch_size))
  n_cols = math.ceil(batch_size / n_rows)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
  axes = axes.flatten()

  for idx, (image, ax) in enumerate(zip(images, axes)):
      image = np.clip(image, 0, 1)
      ax.imshow(image)
      ax.axis("off")

  plt.suptitle(title)
  plt.tight_layout()
  plt.show()

def read_image_from_url(url, height=None, width=None):
  resp = requests.get(url)
  resp.raise_for_status()
  image_bytes = io.BytesIO(resp.content)
  pil_image = Image.open(image_bytes).convert('RGB')
  img_width, img_height = pil_image.size
  if height is not None and width is not None:
    pil_image = pil_image.resize((width, height), Image.BICUBIC)
  return np.float32(pil_image) / 255.

def read_image_from_file(path, height=None, width=None, ext=None):
  if ext is not None:
    path = path.parent / f'{path.stem}.{ext}'
  pil_image = Image.open(path).convert('RGB')
  img_width, img_height = pil_image.size
  if height is not None and width is not None:
    pil_image = pil_image.resize((width, height), Image.BICUBIC)
  return np.float32(pil_image) / 255.

def restore_from_path(path):
  with tf.io.gfile.GFile(path, "rb") as f:
    state = flax.serialization.from_bytes(None, f.read())
  return state

class Bbox(object):
    def __init__(self, top_left_height_width):
       self.top, self.left, self.height, self.width = [
           int(x) for x in top_left_height_width.split('_')
       ]

def draw_image_with_bbox(image, bbox):
  fig, ax = plt.subplots()
  ax.imshow(image)
  rect = patches.Rectangle((bbox.left, bbox.top), bbox.width, bbox.height, linewidth=1, edgecolor='r', facecolor='none')
  ax.add_patch(rect)
  ax.axis("off")
  plt.title("input")
  plt.show()