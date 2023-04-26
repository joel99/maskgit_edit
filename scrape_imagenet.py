#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick scrape for imagenet subset, since the large one is out of scope for me.

JY edits:
1. 224 -> 256 resolution
"""
from pathlib import Path
import os
import json
import tarfile
import subprocess
# import urllib.request
from tqdm import tqdm
import cv2
import numpy as np

from maskgit.notebook_utils import load_label_to_id_and_class_map

OUT_RES = 256
def center_crop(img, new_size):
    h, w, _ = img.shape
    dy = (h - new_size) // 2
    dx = (w - new_size) // 2
    return img[dy:dy + new_size, dx:dx + new_size]

def untar_and_process_images(tar_path, dest_folder):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_folder)

    for img_name in os.listdir(dest_folder):
        if not img_name.endswith(".JPEG"):
            continue
        img_path = Path(dest_folder) / img_name
        img = cv2.imread(str(img_path))
        if img is not None and img.any() and len(img.shape) == 3 and np.mean(img) < 250:
            # resize so min dim is OUT_RES
            h, w, _ = img.shape
            if h < w:
                new_h = OUT_RES
                new_w = int(w * OUT_RES / h)
            else:
                new_w = OUT_RES
                new_h = int(h * OUT_RES / w)
            img = cv2.resize(img, (new_w, new_h))
            img = center_crop(img, OUT_RES)
            save_path = os.path.join(dest_folder, f'proc_{img_name}')
            cv2.imwrite(save_path, img)
        os.remove(img_path)

def download_file(url, output_path):
    # check if exists and reject if so
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return False
    try:
        subprocess.run(["wget", "-O", output_path, url], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return False

def main(requested_labels):
    label_to_id_map, label_to_class_map = load_label_to_id_and_class_map()
    for i, label in enumerate(tqdm(requested_labels, desc="Processing labels")):
        label = label.strip().lower()
        wnid = label_to_id_map[label]
        syn_cls = label_to_class_map[label]
        print(f"preparing class #{i} {label} - {wnid}")
        root = f"./data/imagenet/{syn_cls}"
        os.makedirs(root, exist_ok=True)
        url = f"https://image-net.org/data/winter21_whole/n{wnid}.tar"
        tar_path = f"{root}/{wnid}.tar"
        if not download_file(url, tar_path):
            continue

        untar_and_process_images(tar_path, root)
        os.remove(tar_path)

REQUESTED_LABELS = [
    'pizza',
    'ice cream',
    'eatery',
    'tench',
    'goldfish',
]

if __name__ == "__main__":
    main(REQUESTED_LABELS)
