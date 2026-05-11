import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RECON_W, TV_W, COLOR_W


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w


class ColorStatLoss(nn.Module):
    """Differentiable per-channel mean/variance matching — vectorized across channels."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean(dim=[2, 3])
        target_mean = target.mean(dim=[2, 3])
        pred_var = pred.var(dim=[2, 3], unbiased=False)
        target_var = target.var(dim=[2, 3], unbiased=False)
        return (F.mse_loss(pred_mean, target_mean) + F.mse_loss(pred_var, target_var)) / 3.0


class MixedLoss(nn.Module):
    """Returns raw tensor components — callers decide when to .item() for logging."""

    def __init__(self,
                 recon_w: float = RECON_W,
                 tv_w: float = TV_W,
                 color_w: float = COLOR_W):
        super().__init__()
        self.recon_loss = ReconstructionLoss()
        self.tv_loss = TVLoss()
        self.color_loss = ColorStatLoss()
        self.recon_w = recon_w
        self.tv_w = tv_w
        self.color_w = color_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l_recon = self.recon_loss(pred, target)
        l_tv = self.tv_loss(pred)
        l_color = self.color_loss(pred, target)
        total = self.recon_w * l_recon + self.tv_w * l_tv + self.color_w * l_color
        return total, {
            "recon": l_recon,
            "tv": l_tv,
            "color": l_color,
            "total": total,
        }
