import os

import matplotlib
matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import RESULTS_DIR


def tensor_to_image(tensor: torch.Tensor, to_bgr: bool = True) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.min() < 0:
        img = (img + 1) / 2
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.shape[2] == 3 and to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def calculate_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    if img_a.ndim == 3:
        return ssim(img_a, img_b, channel_axis=2, data_range=255)
    return ssim(img_a, img_b, data_range=255)


def calculate_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return psnr(img_a, img_b, data_range=255)


# Reusable figure for comparison plots to avoid repeated GDI handle allocation on Windows
_cmp_fig = None


def save_comparison_figure(original: np.ndarray,
                           traditional: np.ndarray,
                           deep: np.ndarray,
                           save_path: str):
    global _cmp_fig
    if _cmp_fig is None:
        _cmp_fig, _ = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Original", "Traditional (Cartoon)", "LightUNet (Cartoon)"]
    images = [original, traditional, deep]
    for ax, title, img in zip(_cmp_fig.axes, titles, images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[2] == 3 else img
        ax.clear()
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    _cmp_fig.tight_layout()
    _cmp_fig.savefig(save_path, dpi=150)


def plot_training_curves(losses: list[float],
                         val_psnrs: list[float],
                         save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    epochs = [(i + 1) * 10 for i in range(len(val_psnrs))]
    ax2.plot(epochs, val_psnrs, marker="o")
    ax2.set_title("Validation PSNR")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
