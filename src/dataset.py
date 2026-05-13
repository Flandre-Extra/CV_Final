import os

from torch.utils.data import Dataset

from config import PREPROCESSED_DIR, LABELS_DIR, MAX_PRELOAD_BYTES
from utils import image_to_tensor, imread_any


class StylizationDataset(Dataset):
    """Preloads all image pairs into memory to eliminate per-epoch disk IO."""

    def __init__(self, split: str):
        img_dir = os.path.join(PREPROCESSED_DIR, split)
        label_dir = os.path.join(LABELS_DIR, split)
        files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        )
        self.samples = []
        total_bytes = 0
        warned = False
        for fname in files:
            img = imread_any(os.path.join(img_dir, fname))
            label = imread_any(os.path.join(label_dir, fname))
            if img is None or label is None:
                raise FileNotFoundError(
                    f"Failed to load {fname} from {img_dir} or {label_dir}"
                )
            img_t = image_to_tensor(img)
            label_t = image_to_tensor(label)
            total_bytes += img_t.element_size() * img_t.nelement()
            total_bytes += label_t.element_size() * label_t.nelement()
            if not warned and total_bytes > MAX_PRELOAD_BYTES:
                print(f"WARNING: StylizationDataset[{split}] preload > "
                      f"{MAX_PRELOAD_BYTES / 1024 ** 3:.1f} GB RAM. "
                      f"Consider lazy load or reduce augmentation.")
                warned = True
            self.samples.append((img_t, label_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
