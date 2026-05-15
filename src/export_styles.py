"""Export individual style transfer results for report."""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, RESULTS_DIR, set_seed
from dataset import StylizationDataset
from model import LightUNet
from traditional import cartoonize, pencil_sketch, watercolor
from utils import tensor_to_image


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LightUNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = StylizationDataset("test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    out_dir = os.path.join(RESULTS_DIR, "styles")
    os.makedirs(out_dir, exist_ok=True)

    # Export every Nth sample to avoid 32 nearly-identical outputs
    step = max(1, len(test_ds) // 10)
    exported = 0
    for i, (img_tensor, label_tensor) in enumerate(test_loader):
        if i % step != 0:
            continue

        # Original and label are tensors [0,1], CHW
        img_np = tensor_to_image(img_tensor[0], to_bgr=True)
        label_np = tensor_to_image(label_tensor[0], to_bgr=True)

        # Traditional styles
        cartoon = cartoonize(img_np)
        sketch = pencil_sketch(img_np)
        water = watercolor(img_np)

        # LightUNet
        with torch.no_grad():
            pred = model(img_tensor.to(device))
        dl_cartoon = tensor_to_image(pred[0], to_bgr=True)

        prefix = f"{i:03d}"
        saves = [
            (f"{prefix}_original", img_np),
            (f"{prefix}_trad_cartoon", cartoon),
            (f"{prefix}_trad_sketch", sketch),
            (f"{prefix}_trad_watercolor", water),
            (f"{prefix}_dl_cartoon", dl_cartoon),
        ]
        for name, image in saves:
            path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(path, image)

        print(f"Exported sample {i}: 5 styles")
        exported += 1

    print(f"\nDone. {exported} samples x 5 styles = {exported * 5} images")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
