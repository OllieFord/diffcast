"""Main CSDI model for probabilistic forecasting."""

import torch
import torch.nn as nn
from einops import rearrange

from diffcast.models.csdi.conditioning import ConditioningModule, DiffusionStepEmbedding
from diffcast.models.csdi.diffusion import DiffusionProcess
from diffcast.models.csdi.transformer import DualTransformer, TimeSeriesTransformer


class CSDI(nn.Module):
    """Conditional Score-based Diffusion model for Imputation/Forecasting.

    Adapts the CSDI architecture for day-ahead electricity price forecasting.
    """

    def __init__(
        self,
        n_input_features: int = 10,
        n_covariates: int = 4,
        n_zones: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_diffusion_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        noise_schedule: str = "quadratic",
        context_length: int = 168,
        forecast_length: int = 24,
        multi_zone: bool = False,
    ) -> None:
        """Initialize CSDI model.

        Args:
            n_input_features: Number of input features per timestep
            n_covariates: Number of future known covariates
            n_zones: Number of bidding zones
            d_model: Model hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
            n_diffusion_steps: Number of diffusion steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            noise_schedule: Type of noise schedule
            context_length: Length of historical context
            forecast_length: Length of forecast horizon
            multi_zone: Whether to model all zones jointly
        """
        super().__init__()

        self.n_zones = n_zones
        self.d_model = d_model
        self.context_length = context_length
        self.forecast_length = forecast_length
        self.multi_zone = multi_zone

        # Conditioning module
        self.conditioning = ConditioningModule(
            n_input_features=n_input_features,
            n_covariates=n_covariates,
            n_zones=n_zones,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=2,
            dropout=dropout,
        )

        # Input projection (noisy target + covariates -> d_model)
        # Input is: noisy price (1) + covariate features
        self.input_proj = nn.Linear(1 + d_model, d_model)

        # Transformer backbone
        if multi_zone:
            self.transformer = DualTransformer(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                n_features=n_zones,
                seq_len=forecast_length,
            )
        else:
            self.transformer = TimeSeriesTransformer(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                max_seq_len=forecast_length + context_length,
            )

        # Output projection (d_model -> 1 for noise prediction)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Diffusion process (created separately, not a module)
        self._diffusion: DiffusionProcess | None = None
        self._diffusion_config = {
            "n_steps": n_diffusion_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "schedule": noise_schedule,
        }

    @property
    def diffusion(self) -> DiffusionProcess:
        """Get diffusion process (lazy initialization with correct device)."""
        if self._diffusion is None:
            device = next(self.parameters()).device
            self._diffusion = DiffusionProcess(
                **self._diffusion_config,
                device=device,
            )
        return self._diffusion

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: dict | None = None,
    ) -> torch.Tensor:
        """Forward pass: predict noise from noisy input.

        Args:
            x_t: Noisy target prices (batch, forecast_length) or (batch, n_zones, forecast_length)
            t: Diffusion timesteps (batch,)
            condition: Conditioning dictionary with:
                - context: (batch, context_length, n_features)
                - covariates: (batch, forecast_length, n_covariates)
                - zone_idx: (batch,) zone indices (for single-zone mode)

        Returns:
            Predicted noise (same shape as x_t)
        """
        if condition is None:
            condition = {}

        context = condition.get("context")
        covariates = condition.get("covariates")
        zone_idx = condition.get("zone_idx")

        # Get conditioning signals
        cond = self.conditioning(
            context=context,
            t=t,
            covariates=covariates,
            zone_idx=zone_idx,
        )

        # Reshape input for processing
        if self.multi_zone:
            # x_t: (batch, n_zones, forecast_length)
            batch_size = x_t.shape[0]
            x_t = x_t.unsqueeze(-1)  # (batch, n_zones, forecast_length, 1)

            # Expand covariates for all zones
            if cond["covariate_embed"] is not None:
                cov_embed = cond["covariate_embed"].unsqueeze(1).expand(-1, self.n_zones, -1, -1)
            else:
                cov_embed = torch.zeros(
                    batch_size, self.n_zones, self.forecast_length, self.d_model,
                    device=x_t.device,
                )

            # Concatenate and project
            x = torch.cat([x_t, cov_embed], dim=-1)
            x = self.input_proj(x)  # (batch, n_zones, forecast_length, d_model)

            # Apply timestep conditioning via addition
            time_embed = cond["time_embed"]  # (batch, d_model)
            x = x + time_embed.view(batch_size, 1, 1, -1)

            # Add zone embedding
            if cond["zone_embed"] is not None:
                zone_embed = self.conditioning.zone_embed.weight  # (n_zones, d_model)
                x = x + zone_embed.view(1, self.n_zones, 1, -1)

            # Transformer
            x = self.transformer(x, context=cond["encoded_context"])

            # Output projection
            noise_pred = self.output_proj(x).squeeze(-1)  # (batch, n_zones, forecast_length)

        else:
            # Single zone mode: x_t: (batch, forecast_length)
            batch_size = x_t.shape[0]
            x_t = x_t.unsqueeze(-1)  # (batch, forecast_length, 1)

            # Get covariate embedding
            if cond["covariate_embed"] is not None:
                cov_embed = cond["covariate_embed"]
            else:
                cov_embed = torch.zeros(
                    batch_size, self.forecast_length, self.d_model,
                    device=x_t.device,
                )

            # Concatenate and project
            x = torch.cat([x_t, cov_embed], dim=-1)
            x = self.input_proj(x)  # (batch, forecast_length, d_model)

            # Apply timestep conditioning
            time_embed = cond["time_embed"]
            x = x + time_embed.unsqueeze(1)

            # Add zone embedding
            if cond["zone_embed"] is not None:
                x = x + cond["zone_embed"].unsqueeze(1)

            # Transformer with cross-attention to context
            x = self.transformer(x, context=cond["encoded_context"])

            # Output projection
            noise_pred = self.output_proj(x).squeeze(-1)  # (batch, forecast_length)

        return noise_pred

    def compute_loss(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        covariates: torch.Tensor | None = None,
        zone_idx: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss (noise prediction MSE).

        Args:
            target: Target prices (batch, forecast_length) or (batch, n_zones, forecast_length)
            context: Historical context (batch, context_length, n_features)
            covariates: Future covariates (batch, forecast_length, n_covariates)
            zone_idx: Zone indices (batch,)
            mask: Valid target mask (same shape as target)

        Returns:
            Loss value
        """
        batch_size = target.shape[0]
        device = target.device

        # Sample random timesteps
        t = torch.randint(0, self.diffusion.n_steps, (batch_size,), device=device)

        # Add noise
        x_t, noise = self.diffusion.q_sample(target, t)

        # Predict noise
        condition = {
            "context": context,
            "covariates": covariates,
            "zone_idx": zone_idx,
        }
        noise_pred = self(x_t, t, condition)

        # Compute MSE loss
        loss = (noise_pred - noise) ** 2

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        covariates: torch.Tensor | None = None,
        zone_idx: torch.Tensor | None = None,
        n_samples: int = 100,
        use_ddim: bool = True,
        ddim_steps: int = 20,
    ) -> torch.Tensor:
        """Generate samples from the model.

        Args:
            context: Historical context (batch, context_length, n_features)
            covariates: Future covariates (batch, forecast_length, n_covariates)
            zone_idx: Zone indices (batch,)
            n_samples: Number of samples to generate per input
            use_ddim: Whether to use DDIM sampling
            ddim_steps: Number of DDIM steps

        Returns:
            Samples of shape (batch, n_samples, forecast_length) or
                (batch, n_samples, n_zones, forecast_length)
        """
        batch_size = context.shape[0]
        device = context.device

        # Expand inputs for n_samples
        context_expanded = context.unsqueeze(1).expand(-1, n_samples, -1, -1)
        context_expanded = context_expanded.reshape(batch_size * n_samples, *context.shape[1:])

        if covariates is not None:
            cov_expanded = covariates.unsqueeze(1).expand(-1, n_samples, -1, -1)
            cov_expanded = cov_expanded.reshape(batch_size * n_samples, *covariates.shape[1:])
        else:
            cov_expanded = None

        if zone_idx is not None:
            zone_expanded = zone_idx.unsqueeze(1).expand(-1, n_samples)
            zone_expanded = zone_expanded.reshape(batch_size * n_samples)
        else:
            zone_expanded = None

        condition = {
            "context": context_expanded,
            "covariates": cov_expanded,
            "zone_idx": zone_expanded,
        }

        # Define shape
        if self.multi_zone:
            shape = (batch_size * n_samples, self.n_zones, self.forecast_length)
        else:
            shape = (batch_size * n_samples, self.forecast_length)

        # Sample
        if use_ddim:
            samples = self.diffusion.ddim_sample(
                self,
                shape,
                condition=condition,
                n_steps=ddim_steps,
            )
        else:
            samples = self.diffusion.p_sample_loop(
                self,
                shape,
                condition=condition,
            )

        # Reshape to (batch, n_samples, ...)
        if self.multi_zone:
            samples = samples.reshape(batch_size, n_samples, self.n_zones, self.forecast_length)
        else:
            samples = samples.reshape(batch_size, n_samples, self.forecast_length)

        return samples

    def predict_quantiles(
        self,
        context: torch.Tensor,
        covariates: torch.Tensor | None = None,
        zone_idx: torch.Tensor | None = None,
        quantiles: list[float] | None = None,
        n_samples: int = 100,
        use_ddim: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Generate quantile forecasts.

        Args:
            context: Historical context
            covariates: Future covariates
            zone_idx: Zone indices
            quantiles: Quantile levels (default: [0.1, 0.5, 0.9])
            n_samples: Number of samples for quantile estimation
            use_ddim: Whether to use DDIM sampling

        Returns:
            Dictionary with 'samples' and 'quantiles' tensors
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]

        samples = self.sample(
            context=context,
            covariates=covariates,
            zone_idx=zone_idx,
            n_samples=n_samples,
            use_ddim=use_ddim,
        )

        # Compute quantiles
        quantile_preds = {}
        for q in quantiles:
            quantile_preds[f"q{int(q * 100)}"] = torch.quantile(samples, q, dim=1)

        return {
            "samples": samples,
            "quantiles": quantile_preds,
            "median": torch.median(samples, dim=1).values,
        }
