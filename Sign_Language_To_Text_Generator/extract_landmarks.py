# extract_landmarks.py
# Usage: python extract_landmarks.py --dataset_dir ./dataset --out_csv landmarks.csv

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import mediapipe as mp
from tqdm import tqdm

mp_hands = mp.solutions.hands

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        result = hands.process(img_rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark
        coords = []
        for l in lm:
            coords.append(l.x)
            coords.append(l.y)
            coords.append(l.z)
        return coords  # length 21*3 = 63

def gather_dataset(dataset_dir, out_csv):
    rows = []
    labels = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print("Found label folders:", labels)
    for label in labels:
        folder = os.path.join(dataset_dir, label)
        for fname in tqdm(os.listdir(folder), desc=label):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(folder, fname)
            coords = process_image(path)
            if coords is None:
                # skip if no hand detected
                continue
            row = coords + [label]
            rows.append(row)
    if len(rows) == 0:
        print("No data extracted. Check dataset images and hand visibility.")
        return
    cols = [f"x{i}" for i in range(21) for _ in (0,2)]  # temporary, will overwrite properly below
    # build proper column names
    cols = []
    for i in range(21):
        cols += [f"lx{i}", f"ly{i}", f"lz{i}"]
    cols.append("label")
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root. Subfolders per label.")
    parser.add_argument("--out_csv", type=str, default="landmarks.csv")
    args = parser.parse_args()
    gather_dataset(args.dataset_dir, args.out_csv)
