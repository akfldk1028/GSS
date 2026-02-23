"""Step 03: 3D Gaussian Splatting training (gsplat / 2DGS)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import ClassVar

import numpy as np

from gss.core.step_base import BaseStep
from gss.utils.io import read_colmap_cameras, read_colmap_images, read_colmap_points3d, write_ply
from gss.utils.geometry import make_w2c
from .config import GaussianSplattingConfig
from .contracts import GaussianSplattingInput, GaussianSplattingOutput

logger = logging.getLogger(__name__)


class GaussianSplattingStep(
    BaseStep[GaussianSplattingInput, GaussianSplattingOutput, GaussianSplattingConfig]
):
    name: ClassVar[str] = "gaussian_splatting"
    input_type: ClassVar = GaussianSplattingInput
    output_type: ClassVar = GaussianSplattingOutput
    config_type: ClassVar = GaussianSplattingConfig

    def validate_inputs(self, inputs: GaussianSplattingInput) -> bool:
        if not inputs.frames_dir.exists():
            logger.error(f"Frames directory not found: {inputs.frames_dir}")
            return False
        if not inputs.sparse_dir.exists():
            logger.error(f"COLMAP sparse dir not found: {inputs.sparse_dir}")
            return False
        return True

    def run(self, inputs: GaussianSplattingInput) -> GaussianSplattingOutput:
        import torch
        from gss.utils.losses import ssim_loss

        output_dir = self.data_root / "interim" / "s03_gaussians"
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        # Load COLMAP data
        cameras = read_colmap_cameras(inputs.sparse_dir)
        images = read_colmap_images(inputs.sparse_dir)
        points3d = read_colmap_points3d(inputs.sparse_dir)

        if len(points3d) == 0:
            raise RuntimeError("No 3D points found in COLMAP reconstruction")

        logger.info(
            f"Loaded: {len(cameras)} cameras, {len(images)} images, {len(points3d)} 3D points"
        )

        # Initialize Gaussian parameters from sparse points
        n_points = len(points3d)
        scale_dim = 3  # Always 3 dims; gsplat rasterization_2dgs expects (N, 3)
        params = _init_params(points3d, n_points, scale_dim, device)

        # For 2DGS: set third scale very small (flat disk representation)
        if self.config.method == "2dgs":
            with torch.no_grad():
                params["log_scales"][:, 2] = -10.0  # exp(-10) ≈ 4.5e-5

        # Build optimizer
        optimizer = _build_optimizer(params, self.config.learning_rate)

        # Gradient accumulator for densification
        grad_accum = torch.zeros(n_points, device=device)
        grad_count = torch.zeros(n_points, device=device)

        # Prepare training views
        training_views = self._prepare_views(cameras, images, inputs.frames_dir, device)
        if not training_views:
            raise RuntimeError("No valid training views found")

        # Use standard rasterization for both 2DGS and 3DGS.
        # 2DGS params (flat disk scales) work with standard rasterization.
        # rasterization_2dgs has stricter batch dimension requirements in gsplat >= 1.5.
        from gsplat import rasterization as rasterize_fn

        # Training loop
        n_views = len(training_views)
        rng = np.random.default_rng(42)
        view_order = rng.permutation(n_views)
        cfg = self.config

        for step in range(cfg.iterations):
            if step % n_views == 0:
                view_order = rng.permutation(n_views)
            view = training_views[view_order[step % n_views]]

            # LR scheduling: exponential decay for means
            lr = _exp_lr(cfg.learning_rate, cfg.lr_final, step, cfg.iterations)
            optimizer.param_groups[0]["lr"] = lr

            optimizer.zero_grad()

            scales = torch.exp(params["log_scales"])
            opacities = torch.sigmoid(params["raw_opacities"])

            raster_out = rasterize_fn(
                means=params["means"],
                quats=params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
                scales=scales,
                opacities=opacities,
                colors=params["colors"],
                viewmats=view["viewmat"].unsqueeze(0),
                Ks=view["K"].unsqueeze(0),
                width=view["width"],
                height=view["height"],
            )
            # rasterization returns (renders, alphas, meta)
            renders = raster_out[0]

            # L1 + SSIM loss (lambda_ssim=0.2 per original 3DGS paper Kerbl et al.)
            gt = view["image"].unsqueeze(0)
            loss = ssim_loss(renders, gt, lambda_ssim=0.2)

            loss.backward()

            # Accumulate positional gradients for densification
            if params["means"].grad is not None:
                grad_norm = params["means"].grad.detach().norm(dim=-1)
                grad_accum += grad_norm
                grad_count += 1

            optimizer.step()

            # ── Densification & Pruning ─────────────────────────
            if (
                cfg.densify_from <= step < cfg.densify_until
                and step % cfg.densify_interval == 0
                and step > 0
            ):
                avg_grad = grad_accum / grad_count.clamp(min=1)
                n_before = len(params["means"])

                params, optimizer, grad_accum, grad_count = _densify_and_prune(
                    params=params,
                    optimizer=optimizer,
                    avg_grad=avg_grad,
                    grad_threshold=cfg.densify_grad_threshold,
                    scale_threshold=cfg.densify_scale_threshold,
                    opacity_threshold=cfg.prune_opacity_threshold,
                    max_gaussians=cfg.max_gaussians,
                    scale_dim=scale_dim,
                    device=device,
                )

                n_after = len(params["means"])
                if n_after != n_before:
                    logger.info(
                        f"Step {step}: densify/prune {n_before} -> {n_after} gaussians"
                    )

            # ── Opacity reset ───────────────────────────────────
            if (
                cfg.opacity_reset_interval > 0
                and step % cfg.opacity_reset_interval == 0
                and step > 0
                and step < cfg.densify_until
            ):
                with torch.no_grad():
                    # Reset raw opacities to sigmoid^{-1}(0.01) ≈ -4.595
                    params["raw_opacities"].fill_(-4.595)
                # Reset optimizer state for opacities
                _reset_optimizer_state(optimizer, "opacities")
                logger.info(f"Step {step}: opacity reset")

            if step % 1000 == 0:
                logger.info(
                    f"Step {step}/{cfg.iterations}, loss={loss.item():.4f}, "
                    f"n_gaussians={len(params['means'])}, lr={lr:.2e}"
                )

        # Save result
        model_path = output_dir / "point_cloud.ply"
        final_means = params["means"].detach().cpu().numpy()
        final_colors = torch.sigmoid(params["colors"]).detach().cpu().numpy()
        final_opacities = torch.sigmoid(params["raw_opacities"]).detach().cpu().numpy()
        write_ply(model_path, final_means, final_colors, final_opacities)

        # Torch checkpoint for depth rendering
        checkpoint = {
            "means": params["means"].detach().cpu(),
            "colors": params["colors"].detach().cpu(),
            "log_scales": params["log_scales"].detach().cpu(),
            "quats": params["quats"].detach().cpu(),
            "raw_opacities": params["raw_opacities"].detach().cpu(),
            "method": cfg.method,
        }
        torch.save(checkpoint, output_dir / "model.pt")

        logger.info(
            f"Training complete: {len(final_means)} gaussians, "
            f"{cfg.iterations} iterations"
        )

        return GaussianSplattingOutput(
            model_path=model_path,
            num_gaussians=len(final_means),
            training_iterations=cfg.iterations,
        )

    def _prepare_views(
        self, cameras: dict, images: list[dict], frames_dir: Path, device
    ) -> list[dict]:
        """Convert COLMAP cameras+images to training views (viewmat, K, gt_image)."""
        import torch
        import cv2

        views = []
        for img_data in images:
            cam_id = img_data["camera_id"]
            cam = cameras.get(cam_id)
            if cam is None:
                continue

            img_path = frames_dir / img_data["name"]
            if not img_path.exists():
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt = torch.tensor(rgb, dtype=torch.float32, device=device) / 255.0

            params = cam["params"]
            w = cam["width"]
            h = cam["height"]
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = w / 2, h / 2

            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

            w2c = make_w2c(img_data["qvec"], img_data["tvec"])
            viewmat = torch.tensor(w2c, dtype=torch.float32, device=device)

            views.append({
                "viewmat": viewmat,
                "K": K,
                "image": gt,
                "width": w,
                "height": h,
            })

        logger.info(f"Prepared {len(views)} training views")
        return views


# ── Helper functions ────────────────────────────────────────────────


def _init_params(
    points3d: np.ndarray, n_points: int, scale_dim: int, device
) -> dict:
    """Initialize all Gaussian parameters from sparse SfM points."""
    import torch

    means = torch.tensor(points3d, dtype=torch.float32, device=device, requires_grad=True)
    colors = torch.rand(n_points, 3, dtype=torch.float32, device=device, requires_grad=True)
    log_scales = torch.full(
        (n_points, scale_dim), -3.0,
        dtype=torch.float32, device=device, requires_grad=True,
    )
    quats = torch.zeros(n_points, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)
    raw_opacities = torch.zeros(n_points, dtype=torch.float32, device=device, requires_grad=True)

    return {
        "means": means,
        "colors": colors,
        "log_scales": log_scales,
        "quats": quats,
        "raw_opacities": raw_opacities,
    }


def _build_optimizer(params: dict, means_lr: float):
    """Build Adam optimizer with per-parameter learning rates."""
    import torch
    return torch.optim.Adam([
        {"params": [params["means"]], "lr": means_lr, "name": "means"},
        {"params": [params["colors"]], "lr": 0.0025, "name": "colors"},
        {"params": [params["log_scales"]], "lr": 0.005, "name": "scales"},
        {"params": [params["quats"]], "lr": 0.001, "name": "quats"},
        {"params": [params["raw_opacities"]], "lr": 0.05, "name": "opacities"},
    ])


def _exp_lr(lr_init: float, lr_final: float, step: int, max_steps: int) -> float:
    """Exponential learning rate decay."""
    if max_steps <= 1 or lr_final <= 0 or lr_init <= 0:
        return lr_init
    t = min(step / max_steps, 1.0)
    return lr_init * (lr_final / lr_init) ** t


def _densify_and_prune(
    params: dict,
    optimizer,
    avg_grad,
    grad_threshold: float,
    scale_threshold: float,
    opacity_threshold: float,
    max_gaussians: int,
    scale_dim: int,
    device,
) -> tuple:
    """Adaptive densification (split + clone) and pruning.

    Based on Kerbl et al. 2023, Section 5.2:
    - High gradient + large scale → split into 2 smaller Gaussians
    - High gradient + small scale → clone (duplicate)
    - Low opacity → prune
    """
    import torch

    means = params["means"]
    n = len(means)

    scales = torch.exp(params["log_scales"]).detach()
    opacities = torch.sigmoid(params["raw_opacities"]).detach()
    max_scale = scales.max(dim=-1).values

    # Identify candidates for densification
    high_grad = avg_grad > grad_threshold
    large = max_scale > scale_threshold
    small = ~large

    split_mask = high_grad & large      # split: high grad + large scale
    clone_mask = high_grad & small      # clone: high grad + small scale

    # Don't densify if already at max
    n_new = split_mask.sum().item() + clone_mask.sum().item()
    if n + n_new > max_gaussians:
        clone_mask[:] = False
        split_mask[:] = False

    # ── Split ────────────────────────────────────────────────────
    # Replace each large Gaussian with 2 smaller ones (offset by ±scale)
    split_new = {}
    n_split = split_mask.sum().item()
    if n_split > 0:
        split_means = params["means"].detach()[split_mask]
        split_scales = scales[split_mask]
        # Offset along the direction of maximum scale
        offset_3d = split_scales * 0.5
        split_new["means"] = torch.cat([
            split_means + offset_3d,
            split_means - offset_3d,
        ], dim=0)
        # Halve the scale (in log space: log(s/1.6) = log(s) - log(1.6))
        split_log_scales = params["log_scales"].detach()[split_mask] - math.log(1.6)
        split_new["log_scales"] = split_log_scales.repeat(2, 1)
        split_new["colors"] = params["colors"].detach()[split_mask].repeat(2, 1)
        split_new["quats"] = params["quats"].detach()[split_mask].repeat(2, 1)
        split_new["raw_opacities"] = params["raw_opacities"].detach()[split_mask].repeat(2)

    # ── Clone ────────────────────────────────────────────────────
    clone_new = {}
    n_clone = clone_mask.sum().item()
    if n_clone > 0:
        clone_new["means"] = params["means"].detach()[clone_mask]
        clone_new["log_scales"] = params["log_scales"].detach()[clone_mask]
        clone_new["colors"] = params["colors"].detach()[clone_mask]
        clone_new["quats"] = params["quats"].detach()[clone_mask]
        clone_new["raw_opacities"] = params["raw_opacities"].detach()[clone_mask]

    # ── Prune ────────────────────────────────────────────────────
    # Remove: low opacity OR split sources (they've been replaced)
    keep_mask = (opacities >= opacity_threshold) & (~split_mask)

    # ── Assemble new parameters ──────────────────────────────────
    new_params = {}
    for key in params:
        kept = params[key].detach()[keep_mask]
        parts = [kept]
        if n_split > 0 and key in split_new:
            parts.append(split_new[key])
        if n_clone > 0 and key in clone_new:
            parts.append(clone_new[key])
        new_tensor = torch.cat(parts, dim=0).requires_grad_(True)
        new_params[key] = new_tensor

    # Rebuild optimizer with new tensors
    new_optimizer = _build_optimizer(new_params, optimizer.param_groups[0]["lr"])

    # Reset gradient accumulators
    n_new_total = len(new_params["means"])
    new_grad_accum = torch.zeros(n_new_total, device=device)
    new_grad_count = torch.zeros(n_new_total, device=device)

    return new_params, new_optimizer, new_grad_accum, new_grad_count


def _reset_optimizer_state(optimizer, param_name: str) -> None:
    """Zero out the Adam moment buffers for a specific parameter group."""
    for group in optimizer.param_groups:
        if group.get("name") == param_name:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()
