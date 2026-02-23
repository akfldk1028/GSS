"""Tests for S03: Gaussian Splatting step."""

from pathlib import Path

import numpy as np
import pytest

from gss.steps.s03_gaussian_splatting.config import GaussianSplattingConfig
from gss.steps.s03_gaussian_splatting.contracts import (
    GaussianSplattingInput,
    GaussianSplattingOutput,
)


def _has_torch() -> bool:
    try:
        import torch
        return True
    except (ImportError, OSError):
        return False


class TestGaussianSplattingContracts:
    def test_input_model(self):
        inp = GaussianSplattingInput(
            frames_dir=Path("/tmp/frames"),
            sparse_dir=Path("/tmp/sparse"),
        )
        assert inp.frames_dir == Path("/tmp/frames")

    def test_output_schema(self):
        schema = GaussianSplattingOutput.model_json_schema()
        assert "model_path" in schema["properties"]
        assert "num_gaussians" in schema["properties"]
        assert "training_iterations" in schema["properties"]

    def test_config_defaults(self):
        cfg = GaussianSplattingConfig()
        assert cfg.method == "2dgs"
        assert cfg.iterations == 30000
        assert cfg.densify_from == 500
        assert cfg.densify_until == 15000
        assert cfg.densify_interval == 100
        assert cfg.opacity_reset_interval == 3000

    def test_config_densify_params(self):
        cfg = GaussianSplattingConfig(
            densify_grad_threshold=0.001,
            max_gaussians=100_000,
        )
        assert cfg.densify_grad_threshold == 0.001
        assert cfg.max_gaussians == 100_000


class TestGaussianSplattingStep:
    def test_validate_missing_frames(self, data_root: Path):
        from gss.steps.s03_gaussian_splatting.step import GaussianSplattingStep

        cfg = GaussianSplattingConfig()
        step = GaussianSplattingStep(config=cfg, data_root=data_root)
        inp = GaussianSplattingInput(
            frames_dir=Path("/nonexistent"),
            sparse_dir=Path("/nonexistent"),
        )
        assert step.validate_inputs(inp) is False

    def test_validate_missing_sparse(self, data_root: Path, tmp_path: Path):
        from gss.steps.s03_gaussian_splatting.step import GaussianSplattingStep

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        cfg = GaussianSplattingConfig()
        step = GaussianSplattingStep(config=cfg, data_root=data_root)
        inp = GaussianSplattingInput(
            frames_dir=frames_dir,
            sparse_dir=Path("/nonexistent/sparse"),
        )
        assert step.validate_inputs(inp) is False


class TestExpLR:
    def test_start_equals_init(self):
        from gss.steps.s03_gaussian_splatting.step import _exp_lr
        lr = _exp_lr(1e-3, 1e-6, 0, 30000)
        assert lr == pytest.approx(1e-3)

    def test_end_equals_final(self):
        from gss.steps.s03_gaussian_splatting.step import _exp_lr
        lr = _exp_lr(1e-3, 1e-6, 30000, 30000)
        assert lr == pytest.approx(1e-6, rel=1e-3)

    def test_monotonic_decay(self):
        from gss.steps.s03_gaussian_splatting.step import _exp_lr
        lrs = [_exp_lr(1e-3, 1e-6, s, 30000) for s in range(0, 30001, 5000)]
        for i in range(len(lrs) - 1):
            assert lrs[i] > lrs[i + 1]


@pytest.mark.skipif(not _has_torch(), reason="torch not available")
class TestDensification:
    def test_densify_splits_large_gaussians(self):
        import torch
        from gss.steps.s03_gaussian_splatting.step import (
            _init_params, _build_optimizer, _densify_and_prune,
        )

        device = torch.device("cpu")
        pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        params = _init_params(pts, 3, 3, device)

        # Make one gaussian large (high scale)
        with torch.no_grad():
            params["log_scales"][0] = 2.0  # exp(2) = 7.4, way above threshold

        optimizer = _build_optimizer(params, 1e-4)

        # High gradient for the large gaussian
        avg_grad = torch.tensor([1.0, 0.0, 0.0], device=device)

        new_params, new_opt, _, _ = _densify_and_prune(
            params=params,
            optimizer=optimizer,
            avg_grad=avg_grad,
            grad_threshold=0.0002,
            scale_threshold=0.01,
            opacity_threshold=0.005,
            max_gaussians=500_000,
            scale_dim=3,
            device=device,
        )

        # Gaussian 0 was split: removed + 2 new = net +1
        # Gaussians 1,2 kept (low grad, opacity=0.5 > threshold)
        assert len(new_params["means"]) == 4  # 2 kept + 2 from split

    def test_densify_clones_small_gaussians(self):
        import torch
        from gss.steps.s03_gaussian_splatting.step import (
            _init_params, _build_optimizer, _densify_and_prune,
        )

        device = torch.device("cpu")
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        params = _init_params(pts, 2, 3, device)

        # Both gaussians are small (default log_scale=-3, exp(-3)=0.05 > 0.01)
        # Actually exp(-3) = 0.05, which is > 0.01, so they count as "large"
        # Make them truly small
        with torch.no_grad():
            params["log_scales"].fill_(-7.0)  # exp(-7) ≈ 0.0009

        optimizer = _build_optimizer(params, 1e-4)

        # High gradient for gaussian 0 only
        avg_grad = torch.tensor([1.0, 0.0], device=device)

        new_params, _, _, _ = _densify_and_prune(
            params=params,
            optimizer=optimizer,
            avg_grad=avg_grad,
            grad_threshold=0.0002,
            scale_threshold=0.01,
            opacity_threshold=0.005,
            max_gaussians=500_000,
            scale_dim=3,
            device=device,
        )

        # Gaussian 0: high grad + small scale → clone (+1)
        # Gaussian 1: low grad → keep
        assert len(new_params["means"]) == 3

    def test_prune_low_opacity(self):
        import torch
        from gss.steps.s03_gaussian_splatting.step import (
            _init_params, _build_optimizer, _densify_and_prune,
        )

        device = torch.device("cpu")
        pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
        params = _init_params(pts, 3, 3, device)

        # Make gaussian 1 very low opacity: sigmoid(-10) ≈ 0.00005
        with torch.no_grad():
            params["raw_opacities"][1] = -10.0

        optimizer = _build_optimizer(params, 1e-4)
        avg_grad = torch.zeros(3, device=device)  # no densification

        new_params, _, _, _ = _densify_and_prune(
            params=params,
            optimizer=optimizer,
            avg_grad=avg_grad,
            grad_threshold=0.0002,
            scale_threshold=0.01,
            opacity_threshold=0.005,
            max_gaussians=500_000,
            scale_dim=3,
            device=device,
        )

        # Gaussian 1 pruned
        assert len(new_params["means"]) == 2

    def test_max_gaussians_cap(self):
        import torch
        from gss.steps.s03_gaussian_splatting.step import (
            _init_params, _build_optimizer, _densify_and_prune,
        )

        device = torch.device("cpu")
        pts = np.random.randn(100, 3).astype(np.float32)
        params = _init_params(pts, 100, 3, device)

        optimizer = _build_optimizer(params, 1e-4)
        avg_grad = torch.ones(100, device=device)  # all high grad

        new_params, _, _, _ = _densify_and_prune(
            params=params,
            optimizer=optimizer,
            avg_grad=avg_grad,
            grad_threshold=0.0002,
            scale_threshold=0.01,
            opacity_threshold=0.005,
            max_gaussians=50,  # cap below current count
            scale_dim=3,
            device=device,
        )

        # Should not grow beyond cap (densification skipped)
        assert len(new_params["means"]) <= 100
