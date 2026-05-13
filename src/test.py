import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, RESULTS_DIR, set_seed
from dataset import StylizationDataset
from model import LightUNet
from traditional import cartoonize
from utils import (calculate_psnr, calculate_ssim, save_comparison_figure,
                   tensor_to_image)


def evaluate():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
        if not os.path.exists(checkpoint_path):
            print("ERROR: No checkpoint found (best_model.pth or final_model.pth)")
            print("Run train.py first.")
            sys.exit(1)
        print("Using final_model.pth (no validation was run)")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LightUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_psnr = checkpoint.get("val_psnr", float("nan"))
    print(f"Loaded model from epoch {epoch}, val PSNR: {val_psnr:.2f}")

    test_ds = StylizationDataset("test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    dl_ssim_list, dl_psnr_list = [], []
    trad_ssim_list, trad_psnr_list = [], []
    dl_times, trad_times = [], []

    comparison_dir = os.path.join(RESULTS_DIR, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    if device.type == "cuda" and len(test_ds) > 0:
        warmup_img, _ = test_ds[0]
        warmup_img = warmup_img.unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model(warmup_img)
            torch.cuda.synchronize()

    for i, (img_t, label_t) in enumerate(tqdm(test_loader, desc="Evaluating")):
        img_t = img_t.to(device)
        label_np = tensor_to_image(label_t[0])
        img_np = tensor_to_image(img_t[0])

        with torch.no_grad():
            t0 = time.time()
            pred = model(img_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        dl_time = (t1 - t0) * 1000
        dl_times.append(dl_time)
        pred_np = tensor_to_image(pred[0])
        dl_ssim_list.append(calculate_ssim(pred_np, label_np))
        dl_psnr_list.append(calculate_psnr(pred_np, label_np))

        t0 = time.time()
        trad_np = cartoonize(img_np)
        t1 = time.time()
        trad_time = (t1 - t0) * 1000
        trad_times.append(trad_time)
        trad_ssim_list.append(calculate_ssim(trad_np, label_np))
        trad_psnr_list.append(calculate_psnr(trad_np, label_np))

        if i % 10 == 0:
            save_comparison_figure(img_np, trad_np, pred_np,
                                   os.path.join(comparison_dir, f"comparison_{i:03d}.png"))

    if not dl_ssim_list:
        print("ERROR: No test samples evaluated — check preprocessing output.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Evaluation Results (Test Set)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Traditional':>15} {'LightUNet':>15}")
    print("-" * 50)
    print(f"{'SSIM (mean)':<20} {np.mean(trad_ssim_list):>15.4f} {np.mean(dl_ssim_list):>15.4f}")
    print(f"{'PSNR (mean, dB)':<20} {np.mean(trad_psnr_list):>15.2f} {np.mean(dl_psnr_list):>15.2f}")
    print(f"{'Inference time (ms)':<20} {np.mean(trad_times):>15.2f} {np.mean(dl_times):>15.2f}")
    print("-" * 50)
    print(f"{'Total images evaluated':<20} {len(dl_ssim_list)}")

    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"SSIM:  Traditional={np.mean(trad_ssim_list):.4f}, "
                f"LightUNet={np.mean(dl_ssim_list):.4f}\n")
        f.write(f"PSNR:  Traditional={np.mean(trad_psnr_list):.2f} dB, "
                f"LightUNet={np.mean(dl_psnr_list):.2f} dB\n")
        f.write(f"Speed: Traditional={np.mean(trad_times):.2f} ms, "
                f"LightUNet={np.mean(dl_times):.2f} ms\n")


if __name__ == "__main__":
    evaluate()
