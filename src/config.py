"""Centralized paths and hyperparameters for CV_Final."""

import os
import random

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
PREPROCESSED_DIR = os.path.join(DATASET_DIR, "preprocessed")
LABELS_DIR = os.path.join(DATASET_DIR, "labels_cartoon")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TARGET_SIZE = 512
BLUR_KERNEL = (3, 3)
LAPLACIAN_THRESHOLD = 50
AUG_BRIGHTNESS_FACTORS = [0.85, 1.15]

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 200
LR_STEP_SIZE = 50
LR_GAMMA = 0.5
# Windows 默认 0 避免多进程死锁；Linux / 稳定 Windows 可调 2-4 加速数据加载
NUM_WORKERS = 0

# StylizationDataset 预加载总字节上限（默认 8 GB）。超过则打印警告，提示切换到 lazy load
MAX_PRELOAD_BYTES = 8 * 1024 ** 3

RECON_W = 0.7
TV_W = 0.2
COLOR_W = 0.1

# 早停：连续 N 次验证 PSNR 未提升即停止（每次验证间隔 VAL_INTERVAL 个 epoch）
EARLY_STOP_PATIENCE = 10
VAL_INTERVAL = 10

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cv2.setRNGSeed(seed)


def ensure_dirs() -> None:
    for d in [PREPROCESSED_DIR, LABELS_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    print(f"BASE_DIR:       {BASE_DIR}")
    print(f"RAW_DIR:        {RAW_DIR}")
    print(f"PREPROCESSED_DIR: {PREPROCESSED_DIR}")
    print(f"LABELS_DIR:     {LABELS_DIR}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    print(f"RESULTS_DIR:    {RESULTS_DIR}")
