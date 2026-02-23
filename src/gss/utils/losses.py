"""Loss functions for Gaussian Splatting training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _fspecial_gauss(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a Gaussian kernel (like MATLAB fspecial('gaussian'))."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM between two images.

    Args:
        img1: (B, H, W, C) or (H, W, C) rendered image.
        img2: (B, H, W, C) or (H, W, C) ground truth image.
        window_size: Gaussian window size.
        C1, C2: Stability constants.

    Returns:
        Scalar SSIM value (higher is better, range [0, 1]).
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # (B, H, W, C) -> (B, C, H, W) for conv2d
    img1 = img1.permute(0, 3, 1, 2)
    img2 = img2.permute(0, 3, 1, 2)

    channels = img1.shape[1]
    window = _fspecial_gauss(window_size, 1.5, img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def ssim_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_ssim: float = 0.8,
) -> torch.Tensor:
    """Combined L1 + SSIM loss (original 3DGS paper formulation).

    loss = (1 - λ) * L1 + λ * (1 - SSIM)

    Args:
        rendered: (B, H, W, C) rendered image.
        gt: (B, H, W, C) ground truth image.
        lambda_ssim: Weight for SSIM component (default 0.8 per paper).

    Returns:
        Scalar loss value.
    """
    l1 = (rendered - gt).abs().mean()
    s = ssim(rendered, gt)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - s)
