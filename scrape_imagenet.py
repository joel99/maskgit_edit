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
import urllib.request
from tqdm import tqdm
# import cv2
import numpy as np

OUT_RES = 256
def untar_and_process_images(tar_path, dest_folder):
    import pdb;pdb.set_trace()
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_folder)

    for img_name in os.listdir(dest_folder):
        img_path = Path(dest_folder) / img_name
        img = cv2.imread(img_path)
        if img is not None and img.any() and len(img.shape) == 3 and np.mean(img) < 250:
            img = cv2.resize(img, (OUT_RES, OUT_RES))
            save_path = os.path.join(dest_folder, f'proc_{img_name}.jpg')
            cv2.imwrite(save_path, img)
        os.remove(img_path)

def load_label_to_id_map():
    label_to_id_map = {}
    with open("./data/imagenet_label_synset.txt") as f:
        r"""
            e.g.
            {0: {'id': '01440764-n',
                'label': 'tench, Tinca tinca',
                'uri': 'http://wordnet-rdf.princeton.edu/wn30/01440764-n'},
            1: {'id': '01443537-n',
                'label': 'goldfish, Carassius auratus',
                'uri': 'http://wordnet-rdf.princeton.edu/wn30/01443537-n'},
        """
        # payload = json.load(f) # ! This doesn't work. Need to do standard readlines and construct dict
        # map the individual labels to id
        content = f.read()
        payload = ast.literal_eval(content)
        for info in payload.values():
            for label in info["label"].split(","):
                label_to_id_map[label.strip().lower()] = info["id"][:-2]
    return label_to_id_map

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
        try:
            urllib.request.urlretrieve(url, tar_path)
        except Exception as e:
            print(f"Error downloading tar file for class #{i}: {e}")
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
