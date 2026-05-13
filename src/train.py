import argparse
import json
import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, CHECKPOINT_DIR, EARLY_STOP_PATIENCE, EPOCHS,
                    LEARNING_RATE, LR_GAMMA, LR_STEP_SIZE, NUM_WORKERS,
                    RESULTS_DIR, SEED, VAL_INTERVAL, WEIGHT_DECAY, set_seed)
from dataset import StylizationDataset
from loss import MixedLoss
from model import LightUNet
from utils import calculate_psnr, plot_training_curves, tensor_to_image


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_ds = StylizationDataset("train")
    val_ds = StylizationDataset("val")
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=use_pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=use_pin)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_loader, val_loader


def validate(model: nn.Module,
             dataloader: DataLoader,
             loss_fn: MixedLoss,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0
    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            loss, _ = loss_fn(pred, label)
            batch_samples = img.size(0)
            total_loss += loss.item() * batch_samples
            for i in range(batch_samples):
                p_i = tensor_to_image(pred[i])
                l_i = tensor_to_image(label[i])
                total_psnr += calculate_psnr(p_i, l_i)
            total_samples += batch_samples
    if total_samples == 0:
        raise ValueError("Validation split is empty — check preprocessing output.")
    return total_loss / total_samples, total_psnr / total_samples


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int, avg_loss: float,
                    val_psnr: float, filename: str,
                    save_optimizer: bool = False) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
        "val_psnr": val_psnr,
    }
    if save_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, os.path.join(CHECKPOINT_DIR, filename))


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    loss_fn: MixedLoss,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler | None,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int) -> float:
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    use_amp = device.type == "cuda"
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{total_epochs}")
    for img, label in pbar:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                pred = model(img)
                loss, components = loss_fn(pred, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(img)
            loss, components = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
        batch_samples = img.size(0)
        epoch_loss += loss.item() * batch_samples
        total_samples += batch_samples
        pbar.set_postfix({
            "loss": f"{components['total'].item():.4f}",
            "recon": f"{components['recon'].item():.4f}",
        })
    return epoch_loss / total_samples


def train(epochs: int = EPOCHS):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, val_loader = build_dataloaders()

    model = LightUNet().to(device)
    loss_fn = MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE,
                                                gamma=LR_GAMMA)
    scaler = GradScaler() if device.type == "cuda" else None

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    losses = []
    val_psnrs = []
    best_psnr = float("-inf")
    no_improve = 0

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                   scaler, device, epoch + 1, epochs)
        losses.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % VAL_INTERVAL == 0:
            val_loss, val_psnr = validate(model, val_loader, loss_fn, device)
            val_psnrs.append(val_psnr)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")

            save_checkpoint(model, optimizer, epoch + 1, avg_loss, val_psnr,
                            f"model_epoch_{epoch+1:03d}.pth")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                no_improve = 0
                save_checkpoint(model, optimizer, epoch + 1, avg_loss, val_psnr,
                                "best_model.pth", save_optimizer=True)
                print(f"  New best PSNR: {best_psnr:.2f} dB")
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve}/{EARLY_STOP_PATIENCE} checks")

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (PSNR stalled for "
                      f"{no_improve * VAL_INTERVAL} epochs)")
                break

    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(CHECKPOINT_DIR, "final_model.pth"))

    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({"losses": losses, "val_psnrs": val_psnrs}, f)

    plot_training_curves(losses, val_psnrs,
                         os.path.join(RESULTS_DIR, "training_curves.png"))
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightUNet for image stylization")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Override training epochs (default: {EPOCHS})")
    args = parser.parse_args()
    train(epochs=args.epochs)
