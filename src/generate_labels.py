import os
import sys

from tqdm import tqdm

from config import LABELS_DIR, PREPROCESSED_DIR, set_seed
from traditional import cartoonize
from utils import imread_any, imwrite_any


def generate_labels():
    set_seed()
    any_processed = False
    for split in ["train", "val", "test"]:
        src_dir = os.path.join(PREPROCESSED_DIR, split)
        dst_dir = os.path.join(LABELS_DIR, split)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"WARNING: {src_dir} does not exist. Run preprocess.py first.")
            continue

        files = [f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))]
        if files:
            any_processed = True
        for fname in tqdm(files, desc=f"Generating labels for {split}"):
            src_path = os.path.join(src_dir, fname)
            img = imread_any(src_path)
            if img is None:
                print(f"WARNING: Failed to load {src_path}, skipping")
                continue
            cartoon = cartoonize(img)
            if not imwrite_any(os.path.join(dst_dir, fname), cartoon):
                raise IOError(f"Failed to write label: {dst_dir}/{fname}")

        print(f"  {split}: {len(files)} labels generated")

    if not any_processed:
        print("ERROR: No preprocessed splits found. Run preprocess.py first.")
        sys.exit(1)


if __name__ == "__main__":
    generate_labels()
