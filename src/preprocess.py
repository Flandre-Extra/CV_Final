import argparse
import os
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (AUG_BRIGHTNESS_FACTORS, BLUR_KERNEL, LAPLACIAN_THRESHOLD,
                    PREPROCESSED_DIR, RAW_DIR, SEED, TARGET_SIZE, set_seed)
from utils import imread_any, imwrite_any


def collect_images(raw_dir: str) -> list[tuple[str, str]]:
    items = []
    for category in os.listdir(raw_dir):
        cat_path = os.path.join(raw_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((category, os.path.join(cat_path, fname)))
    return items


def is_sharp(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var >= LAPLACIAN_THRESHOLD


def preprocess_one(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    image = cv2.GaussianBlur(image, BLUR_KERNEL, 0)
    return image


def augment(image: np.ndarray) -> list[np.ndarray]:
    results = [image]
    results.append(cv2.flip(image, 1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    for factor in AUG_BRIGHTNESS_FACTORS:
        hsv_aug = hsv.copy()
        hsv_aug[:, :, 2] = np.clip(hsv_aug[:, :, 2] * factor, 0, 255)
        results.append(cv2.cvtColor(hsv_aug.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return results


def main(max_total: int | None = None):
    set_seed()
    items = collect_images(RAW_DIR)
    print(f"Found {len(items)} raw images")

    if max_total and len(items) > max_total:
        random.shuffle(items)
        items = items[:max_total]
        print(f"  Smoke test mode: limited to {max_total} images")

    # Single pass: read + sharp-check + preprocess, keep arrays to avoid 2x imread
    sharp_items = []
    rejected = 0
    for category, path in tqdm(items, desc="Checking sharpness"):
        img = imread_any(path)
        if img is None:
            print(f"WARNING: Failed to load {path}, skipping")
            continue
        if is_sharp(img):
            img = preprocess_one(img)
            sharp_items.append((category, img))
        else:
            rejected += 1
    print(f"Sharp: {len(sharp_items)}, Rejected (blurry): {rejected}")

    if len(sharp_items) < 200:
        print(f"WARNING: Only {len(sharp_items)} sharp images. Need 300+ for good results.")
        print("Consider taking more photos or lowering LAPLACIAN_THRESHOLD.")
        if max_total:
            print("(Ignorable — smoke test with reduced dataset)")

    train_items, val_items, test_items = [], [], []
    categories = set(c for c, _ in sharp_items)
    for cat in categories:
        cat_items = [(c, img) for c, img in sharp_items if c == cat]
        cat_train, cat_tmp = train_test_split(cat_items, test_size=0.30, random_state=SEED)
        cat_val, cat_test = train_test_split(cat_tmp, test_size=0.50, random_state=SEED)
        train_items.extend(cat_train)
        val_items.extend(cat_val)
        test_items.extend(cat_test)

    print(f"Split: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    for split_name, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        split_dir = os.path.join(PREPROCESSED_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        count = 0
        for category, img in tqdm(split_items, desc=f"Processing {split_name}"):
            images = augment(img) if split_name == "train" else [img]
            for aug_img in images:
                fname = f"{category}_{count:04d}.jpg"
                imwrite_any(os.path.join(split_dir, fname), aug_img)
                count += 1
        print(f"  {split_name}: {count} images saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw images for CV_Final")
    parser.add_argument("--max-total", type=int, default=None,
                        help="Limit total images (smoke test: 32)")
    args = parser.parse_args()
    main(max_total=args.max_total)
