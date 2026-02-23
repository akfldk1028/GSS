"""Tests for loss functions."""

import pytest


def _has_torch() -> bool:
    try:
        import torch
        return True
    except (ImportError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _has_torch(), reason="torch not available")


class TestSSIM:
    def test_identical_images(self):
        """SSIM of identical images should be ~1.0."""
        import torch
        from gss.utils.losses import ssim
        img = torch.rand(1, 32, 32, 3)
        result = ssim(img, img)
        assert result.item() == pytest.approx(1.0, abs=1e-4)

    def test_different_images(self):
        """SSIM of very different images should be low."""
        import torch
        from gss.utils.losses import ssim
        img1 = torch.zeros(1, 32, 32, 3)
        img2 = torch.ones(1, 32, 32, 3)
        result = ssim(img1, img2)
        assert result.item() < 0.1

    def test_output_range(self):
        """SSIM should be in [-1, 1]."""
        import torch
        from gss.utils.losses import ssim
        img1 = torch.rand(1, 32, 32, 3)
        img2 = torch.rand(1, 32, 32, 3)
        result = ssim(img1, img2)
        assert -1.0 <= result.item() <= 1.0

    def test_3d_input(self):
        """Should accept (H, W, C) input without batch dim."""
        import torch
        from gss.utils.losses import ssim
        img = torch.rand(32, 32, 3)
        result = ssim(img, img)
        assert result.item() == pytest.approx(1.0, abs=1e-4)


class TestSSIMLoss:
    def test_identical_loss_zero(self):
        """Loss should be ~0 for identical images."""
        import torch
        from gss.utils.losses import ssim_loss
        img = torch.rand(1, 32, 32, 3)
        loss = ssim_loss(img, img)
        assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_different_loss_high(self):
        """Loss should be high for very different images."""
        import torch
        from gss.utils.losses import ssim_loss
        img1 = torch.zeros(1, 32, 32, 3)
        img2 = torch.ones(1, 32, 32, 3)
        loss = ssim_loss(img1, img2)
        assert loss.item() > 0.5

    def test_gradient_flows(self):
        """Loss should support backpropagation."""
        import torch
        from gss.utils.losses import ssim_loss
        rendered = torch.rand(1, 32, 32, 3, requires_grad=True)
        gt = torch.rand(1, 32, 32, 3)
        loss = ssim_loss(rendered, gt)
        loss.backward()
        assert rendered.grad is not None
        assert rendered.grad.shape == rendered.shape

    def test_lambda_zero_is_l1(self):
        """With lambda_ssim=0, loss should be pure L1."""
        import torch
        from gss.utils.losses import ssim_loss
        img1 = torch.rand(1, 32, 32, 3)
        img2 = torch.rand(1, 32, 32, 3)
        loss_combined = ssim_loss(img1, img2, lambda_ssim=0.0)
        l1_loss = (img1 - img2).abs().mean()
        assert loss_combined.item() == pytest.approx(l1_loss.item(), abs=1e-5)
