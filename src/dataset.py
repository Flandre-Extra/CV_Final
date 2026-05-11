import os

import cv2
from torch.utils.data import Dataset

from config import PREPROCESSED_DIR, LABELS_DIR
from utils import image_to_tensor


class StylizationDataset(Dataset):
    """Preloads all image pairs into memory to eliminate per-epoch disk IO."""

    def __init__(self, split: str):
        img_dir = os.path.join(PREPROCESSED_DIR, split)
        label_dir = os.path.join(LABELS_DIR, split)
        files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        )
        self.samples = []
        for fname in files:
            img = cv2.imread(os.path.join(img_dir, fname))
            label = cv2.imread(os.path.join(label_dir, fname))
            if img is None or label is None:
                raise FileNotFoundError(
                    f"Failed to load {fname} from {img_dir} or {label_dir}"
                )
            self.samples.append((image_to_tensor(img), image_to_tensor(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
