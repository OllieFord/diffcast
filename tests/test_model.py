"""Tests for CSDI model components."""

import pytest
import torch

from diffcast.models.csdi import CSDI
from diffcast.models.csdi.diffusion import DiffusionProcess
from diffcast.models.csdi.transformer import DualTransformer, TimeSeriesTransformer


class TestDiffusionProcess:
    """Tests for diffusion process."""

    def test_forward_diffusion(self):
        """Test forward diffusion adds noise correctly."""
        diffusion = DiffusionProcess(n_steps=50, schedule="quadratic")

        x_0 = torch.randn(4, 24)  # batch=4, seq_len=24
        t = torch.tensor([0, 10, 25, 49])

        x_t, noise = diffusion.q_sample(x_0, t)

        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape

    def test_noise_increases_with_timestep(self):
        """Test that noise level increases with timestep."""
        diffusion = DiffusionProcess(n_steps=50)

        x_0 = torch.zeros(2, 24)
        noise = torch.ones(2, 24)

        t_low = torch.tensor([0, 0])
        t_high = torch.tensor([49, 49])

        x_t_low, _ = diffusion.q_sample(x_0, t_low, noise=noise)
        x_t_high, _ = diffusion.q_sample(x_0, t_high, noise=noise)

        # Higher timestep should have more noise
        assert x_t_high.abs().mean() > x_t_low.abs().mean()

    def test_predict_x0_from_noise(self):
        """Test reconstruction of x_0 from noise prediction."""
        diffusion = DiffusionProcess(n_steps=50)

        x_0 = torch.randn(4, 24)
        t = torch.tensor([10, 10, 10, 10])

        x_t, noise = diffusion.q_sample(x_0, t)
        x_0_pred = diffusion.predict_x0_from_noise(x_t, t, noise)

        # Should recover x_0 perfectly with true noise
        assert torch.allclose(x_0_pred, x_0, atol=1e-5)


class TestTransformer:
    """Tests for transformer components."""

    def test_time_series_transformer_forward(self):
        """Test TimeSeriesTransformer forward pass."""
        model = TimeSeriesTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
        )

        x = torch.randn(4, 24, 64)  # batch=4, seq_len=24, d_model=64
        context = torch.randn(4, 168, 64)  # context from encoder

        out = model(x, context=context)

        assert out.shape == x.shape

    def test_dual_transformer_forward(self):
        """Test DualTransformer forward pass."""
        model = DualTransformer(
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_features=10,  # zones
            seq_len=24,
        )

        x = torch.randn(4, 10, 24, 64)  # batch=4, zones=10, seq_len=24, d_model=64
        context = torch.randn(4, 168, 64)

        out = model(x, context=context)

        assert out.shape == x.shape


class TestCSDI:
    """Tests for full CSDI model."""

    @pytest.fixture
    def model(self):
        """Create model for testing."""
        return CSDI(
            n_input_features=10,
            n_covariates=4,
            n_zones=10,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_diffusion_steps=10,
            context_length=168,
            forecast_length=24,
            multi_zone=False,
        )

    def test_forward_pass(self, model):
        """Test forward pass produces correct shape."""
        batch_size = 4

        x_t = torch.randn(batch_size, 24)
        t = torch.randint(0, 10, (batch_size,))
        condition = {
            "context": torch.randn(batch_size, 168, 10),
            "covariates": torch.randn(batch_size, 24, 4),
            "zone_idx": torch.randint(0, 10, (batch_size,)),
        }

        noise_pred = model(x_t, t, condition)

        assert noise_pred.shape == x_t.shape

    def test_compute_loss(self, model):
        """Test loss computation."""
        batch_size = 4

        target = torch.randn(batch_size, 24)
        context = torch.randn(batch_size, 168, 10)
        covariates = torch.randn(batch_size, 24, 4)
        zone_idx = torch.randint(0, 10, (batch_size,))

        loss = model.compute_loss(
            target=target,
            context=context,
            covariates=covariates,
            zone_idx=zone_idx,
        )

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Positive loss

    def test_sample(self, model):
        """Test sample generation."""
        batch_size = 2
        n_samples = 5

        context = torch.randn(batch_size, 168, 10)
        covariates = torch.randn(batch_size, 24, 4)
        zone_idx = torch.randint(0, 10, (batch_size,))

        samples = model.sample(
            context=context,
            covariates=covariates,
            zone_idx=zone_idx,
            n_samples=n_samples,
            use_ddim=True,
            ddim_steps=5,
        )

        assert samples.shape == (batch_size, n_samples, 24)


class TestCSDIMultiZone:
    """Tests for multi-zone CSDI model."""

    @pytest.fixture
    def model(self):
        """Create multi-zone model for testing."""
        return CSDI(
            n_input_features=10,
            n_covariates=4,
            n_zones=10,
            d_model=64,
            n_heads=4,
            n_layers=2,
            n_diffusion_steps=10,
            context_length=168,
            forecast_length=24,
            multi_zone=True,
        )

    def test_forward_pass_multi_zone(self, model):
        """Test multi-zone forward pass."""
        batch_size = 4

        x_t = torch.randn(batch_size, 10, 24)  # (batch, zones, seq_len)
        t = torch.randint(0, 10, (batch_size,))
        condition = {
            "context": torch.randn(batch_size, 168, 10),
            "covariates": torch.randn(batch_size, 24, 4),
        }

        noise_pred = model(x_t, t, condition)

        assert noise_pred.shape == x_t.shape

    def test_sample_multi_zone(self, model):
        """Test multi-zone sampling."""
        batch_size = 2
        n_samples = 3

        context = torch.randn(batch_size, 168, 10)

        samples = model.sample(
            context=context,
            n_samples=n_samples,
            use_ddim=True,
            ddim_steps=5,
        )

        assert samples.shape == (batch_size, n_samples, 10, 24)
