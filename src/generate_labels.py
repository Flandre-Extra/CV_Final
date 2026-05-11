import os

import cv2
from tqdm import tqdm

from config import LABELS_DIR, PREPROCESSED_DIR, set_seed
from traditional import cartoonize


def generate_labels():
    set_seed()
    for split in ["train", "val", "test"]:
        src_dir = os.path.join(PREPROCESSED_DIR, split)
        dst_dir = os.path.join(LABELS_DIR, split)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"WARNING: {src_dir} does not exist. Run preprocess.py first.")
            continue

        files = [f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))]
        for fname in tqdm(files, desc=f"Generating labels for {split}"):
            src_path = os.path.join(src_dir, fname)
            img = cv2.imread(src_path)
            if img is None:
                print(f"WARNING: Failed to load {src_path}, skipping")
                continue
            cartoon = cartoonize(img)
            cv2.imwrite(os.path.join(dst_dir, fname), cartoon)

        print(f"  {split}: {len(files)} labels generated")


if __name__ == "__main__":
    generate_labels()
