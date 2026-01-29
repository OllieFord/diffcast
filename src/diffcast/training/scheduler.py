"""Noise schedules for diffusion models."""

import math

import torch


class NoiseScheduler:
    """Noise scheduler with various schedule types."""

    def __init__(
        self,
        n_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "quadratic",
    ) -> None:
        """Initialize noise scheduler.

        Args:
            n_steps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule: Schedule type ('linear', 'quadratic', 'cosine', 'sigmoid')
        """
        self.n_steps = n_steps

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, n_steps)
        elif schedule == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, n_steps) ** 2
        elif schedule == "cosine":
            self.betas = self._cosine_schedule(n_steps)
        elif schedule == "sigmoid":
            self.betas = self._sigmoid_schedule(n_steps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def _cosine_schedule(self, n_steps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved DDPM'."""
        steps = torch.linspace(0, n_steps, n_steps + 1)
        alphas_cumprod = torch.cos(((steps / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _sigmoid_schedule(
        self,
        n_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Sigmoid schedule for smoother transitions."""
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        return betas

    def get_snr(self, t: int) -> float:
        """Get signal-to-noise ratio at timestep t."""
        alpha_cumprod = self.alphas_cumprod[t]
        return alpha_cumprod / (1 - alpha_cumprod)

    def get_noise_level(self, t: int) -> tuple[float, float]:
        """Get noise level components at timestep t.

        Returns:
            Tuple of (sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod)
        """
        alpha_cumprod = self.alphas_cumprod[t]
        return (
            math.sqrt(alpha_cumprod.item()),
            math.sqrt(1 - alpha_cumprod.item()),
        )


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        """Initialize scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> None:
        """Update learning rate."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
