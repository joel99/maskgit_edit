#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick scrape for imagenet subset, since the large one is out of scope for me.

JY edits:
1. 224 -> 256 resolution
"""
import ast
from pathlib import Path
import os
import json
import tarfile
import subprocess
# import urllib.request
from tqdm import tqdm
import cv2
import numpy as np

OUT_RES = 256
def center_crop(img, new_size):
    h, w, _ = img.shape
    dy = (h - new_size) // 2
    dx = (w - new_size) // 2
    return img[dy:dy + new_size, dx:dx + new_size]

def untar_and_process_images(tar_path, dest_folder, out_dir="proc"):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_folder)

    if not os.path.exists(os.path.join(dest_folder, out_dir)):
        os.makedirs(os.path.join(dest_folder, out_dir))

    for img_name in os.listdir(dest_folder):
        img_path = Path(dest_folder) / img_name
        img = cv2.imread(str(img_path))
        if img is not None and img.any() and len(img.shape) == 3 and np.mean(img) < 250:
            img = center_crop(img, (OUT_RES, OUT_RES))
            save_path = os.path.join(dest_folder, 'proc', img_name)
            cv2.imwrite(save_path, img)
        os.remove(img_path)

def load_label_to_id_map():
    label_to_id_map = {}
    with open("./resources/imagenet_label_synset.txt") as f:
        r"""
            e.g.
            {0: {'id': '01440764-n',
                'label': 'tench, Tinca tinca',
                'uri': 'http://wordnet-rdf.princeton.edu/wn30/01440764-n'},
            1: {'id': '01443537-n',
                'label': 'goldfish, Carassius auratus',
                'uri': 'http://wordnet-rdf.princeton.edu/wn30/01443537-n'},
        """
        content = f.read()
        payload = ast.literal_eval(content)
        for info in payload.values():
            for label in info["label"].split(","):
                label_to_id_map[label.strip().lower()] = info["id"][:-2]
    return label_to_id_map

def download_file(url, output_path):
    try:
        subprocess.run(["wget", "-O", output_path, url], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        return False

def main(requested_labels):
    label_to_id_map = load_label_to_id_map()
    for i, label in enumerate(tqdm(requested_labels, desc="Processing labels")):
        label = label.strip().lower()
        wnid = label_to_id_map[label]
        print(f"preparing class #{i} {label} - {wnid}")
        root = f"./data/imagenet/{wnid}"
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
