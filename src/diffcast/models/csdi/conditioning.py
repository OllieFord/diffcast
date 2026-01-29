"""Conditioning mechanisms for CSDI model."""

import torch
import torch.nn as nn
from einops import rearrange

from diffcast.models.csdi.transformer import MultiHeadAttention, SinusoidalPositionEmbeddings


class HistoryEncoder(nn.Module):
    """Encoder for historical context (conditioning signal)."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize history encoder.

        Args:
            input_dim: Number of input features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict({
                    "attn": MultiHeadAttention(d_model, n_heads, dropout),
                    "norm1": nn.LayerNorm(d_model),
                    "ff": nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model * 4, d_model),
                        nn.Dropout(dropout),
                    ),
                    "norm2": nn.LayerNorm(d_model),
                })
            )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode historical context.

        Args:
            x: Historical features of shape (batch, context_len, n_features)

        Returns:
            Encoded context of shape (batch, context_len, d_model)
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer["attn"](layer["norm1"](x))
            x = x + layer["ff"](layer["norm2"](x))

        return self.output_norm(x)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for conditioning.

    Applies affine transformation based on conditioning signal:
    output = gamma * x + beta
    """

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
    ) -> None:
        """Initialize FiLM layer.

        Args:
            d_model: Dimension of features to modulate
            cond_dim: Dimension of conditioning signal
        """
        super().__init__()
        self.gamma_proj = nn.Linear(cond_dim, d_model)
        self.beta_proj = nn.Linear(cond_dim, d_model)

        # Initialize to identity transform
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Features to modulate (batch, ..., d_model)
            cond: Conditioning signal (batch, cond_dim)

        Returns:
            Modulated features (same shape as x)
        """
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)

        # Reshape for broadcasting
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


class DiffusionStepEmbedding(nn.Module):
    """Embedding for diffusion timestep."""

    def __init__(
        self,
        d_model: int,
        max_steps: int = 1000,
    ) -> None:
        """Initialize timestep embedding.

        Args:
            d_model: Output dimension
            max_steps: Maximum number of diffusion steps
        """
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute timestep embedding.

        Args:
            t: Timestep indices of shape (batch,)

        Returns:
            Embeddings of shape (batch, d_model)
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class CovariateEncoder(nn.Module):
    """Encoder for future known covariates (calendar, weather forecasts)."""

    def __init__(
        self,
        n_covariates: int,
        d_model: int,
    ) -> None:
        """Initialize covariate encoder.

        Args:
            n_covariates: Number of covariate features
            d_model: Output dimension
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_covariates, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, covariates: torch.Tensor) -> torch.Tensor:
        """Encode covariates.

        Args:
            covariates: Future covariates of shape (batch, forecast_len, n_covariates)

        Returns:
            Encoded covariates of shape (batch, forecast_len, d_model)
        """
        return self.proj(covariates)


class ConditioningModule(nn.Module):
    """Combined conditioning module for CSDI.

    Combines:
    1. Encoded historical context (cross-attention)
    2. Diffusion timestep embedding
    3. Future known covariates
    4. Zone embeddings
    """

    def __init__(
        self,
        n_input_features: int = 10,
        n_covariates: int = 4,
        n_zones: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize conditioning module.

        Args:
            n_input_features: Number of input features
            n_covariates: Number of future covariates
            n_zones: Number of bidding zones
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # History encoder
        self.history_encoder = HistoryEncoder(
            input_dim=n_input_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dropout=dropout,
        )

        # Timestep embedding
        self.time_embed = DiffusionStepEmbedding(d_model)

        # Covariate encoder
        self.covariate_encoder = CovariateEncoder(n_covariates, d_model)

        # Zone embedding
        self.zone_embed = nn.Embedding(n_zones, d_model)

        # FiLM layers for timestep conditioning
        self.film = FiLMConditioning(d_model, d_model)

    def forward(
        self,
        context: torch.Tensor,
        t: torch.Tensor,
        covariates: torch.Tensor | None = None,
        zone_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all conditioning signals.

        Args:
            context: Historical features (batch, context_len, n_features)
            t: Diffusion timestep (batch,)
            covariates: Future covariates (batch, forecast_len, n_covariates)
            zone_idx: Zone indices (batch,)

        Returns:
            Dictionary with conditioning tensors:
                - encoded_context: (batch, context_len, d_model)
                - time_embed: (batch, d_model)
                - covariate_embed: (batch, forecast_len, d_model)
                - zone_embed: (batch, d_model)
        """
        result = {}

        # Encode history
        result["encoded_context"] = self.history_encoder(context)

        # Timestep embedding
        result["time_embed"] = self.time_embed(t)

        # Covariate embedding
        if covariates is not None:
            result["covariate_embed"] = self.covariate_encoder(covariates)
        else:
            result["covariate_embed"] = None

        # Zone embedding
        if zone_idx is not None:
            result["zone_embed"] = self.zone_embed(zone_idx)
        else:
            result["zone_embed"] = None

        return result

    def apply_film(self, x: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning based on timestep.

        Args:
            x: Features to modulate
            time_embed: Timestep embedding

        Returns:
            Modulated features
        """
        return self.film(x, time_embed)
