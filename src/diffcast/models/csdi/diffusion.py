"""Diffusion process for CSDI model."""

import math

import torch
import torch.nn as nn


class DiffusionProcess:
    """Discrete diffusion process with forward and reverse operations.

    Implements the DDPM forward process q(x_t | x_0) and provides
    utilities for the reverse denoising process.
    """

    def __init__(
        self,
        n_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "quadratic",
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize diffusion process.

        Args:
            n_steps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: Noise schedule type ('linear', 'quadratic', 'cosine')
            device: Device for tensors
        """
        self.n_steps = n_steps
        self.device = device

        # Compute beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_steps) ** 2
        elif schedule == "cosine":
            steps = torch.linspace(0, n_steps, n_steps + 1)
            alphas_cumprod = torch.cos(((steps / n_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance for q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        # Coefficients for posterior mean
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract coefficient at timestep t and reshape for broadcasting.

        Args:
            a: Coefficient tensor of shape (n_steps,)
            t: Timestep indices of shape (batch,)
            x_shape: Shape of x for broadcasting

        Returns:
            Extracted values reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from forward diffusion process q(x_t | x_0).

        Args:
            x_0: Clean data of shape (batch, ...)
            t: Timestep indices of shape (batch,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_0: Clean data prediction
            x_t: Noisy data at time t
            t: Timestep indices

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise.

        Args:
            x_t: Noisy data at time t
            t: Timestep indices
            noise: Predicted noise

        Returns:
            Predicted x_0
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Single reverse diffusion step p(x_{t-1} | x_t).

        Args:
            model: Noise prediction model
            x_t: Noisy data at time t
            t: Timestep indices
            condition: Conditioning information
            clip_denoised: Whether to clip predicted x_0

        Returns:
            Denoised sample x_{t-1}
        """
        # Predict noise
        noise_pred = model(x_t, t, condition)

        # Predict x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        if clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)  # Normalized price range

        # Get posterior mean and variance
        posterior_mean, posterior_variance, _ = self.q_posterior_mean_variance(x_0_pred, x_t, t)

        # Sample
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        condition: dict | None = None,
        clip_denoised: bool = True,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Full reverse diffusion sampling loop.

        Args:
            model: Noise prediction model
            shape: Shape of samples to generate
            condition: Conditioning information
            clip_denoised: Whether to clip predictions
            return_intermediates: Whether to return all intermediate steps

        Returns:
            Generated samples (and optionally intermediates)
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        x = torch.randn(shape, device=device)
        intermediates = [x] if return_intermediates else []

        for i in reversed(range(self.n_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition, clip_denoised)
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        condition: dict | None = None,
        n_steps: int = 20,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling for faster inference.

        Args:
            model: Noise prediction model
            shape: Shape of samples to generate
            condition: Conditioning information
            n_steps: Number of DDIM steps (can be less than training steps)
            eta: Stochasticity parameter (0 = deterministic)
            clip_denoised: Whether to clip predictions

        Returns:
            Generated samples
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Create sub-sequence of timesteps
        step_size = self.n_steps // n_steps
        timesteps = list(range(0, self.n_steps, step_size))

        # Start from noise
        x = torch.randn(shape, device=device)

        for i in reversed(range(len(timesteps))):
            t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)

            # Predict noise
            noise_pred = model(x, t, condition)

            # Predict x_0
            alpha_t = self._extract(self.alphas_cumprod, t, x.shape)
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            if clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)

            if i > 0:
                # Get alpha for previous timestep
                t_prev = torch.full(
                    (batch_size,), timesteps[i - 1], device=device, dtype=torch.long
                )
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)

                # DDIM update
                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t))
                    * torch.sqrt(1 - alpha_t / alpha_t_prev)
                )

                pred_dir = torch.sqrt(1 - alpha_t_prev - sigma**2) * noise_pred
                noise = torch.randn_like(x) if eta > 0 else 0

                x = torch.sqrt(alpha_t_prev) * x_0_pred + pred_dir + sigma * noise
            else:
                x = x_0_pred

        return x
