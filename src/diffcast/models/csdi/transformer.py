"""Dual transformer backbone for CSDI."""

import math

import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings.

        Args:
            t: Timestep indices of shape (batch,)

        Returns:
            Embeddings of shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query input of shape (batch, seq_len, d_model)
            context: Key/value input (defaults to x for self-attention)
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        if context is None:
            context = x

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.norm_cross = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            context: Optional context for cross-attention
            mask: Optional attention mask

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), mask=mask)

        # Cross-attention (if context provided)
        if self.cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context)

        # Feed-forward
        x = x + self.ff(self.norm2(x))

        return x


class DualTransformer(nn.Module):
    """Dual transformer with alternating temporal and feature/zone attention.

    Following the CSDI paper, this alternates between:
    1. Temporal attention: Each feature attends across time
    2. Feature attention: Each timestep attends across features/zones
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        n_features: int = 10,
        seq_len: int = 24,
    ) -> None:
        """Initialize dual transformer.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers (each has temporal + feature attn)
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            n_features: Number of features/zones
            seq_len: Sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_features = n_features
        self.seq_len = seq_len

        # Temporal transformer blocks
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, cross_attention=True)
            for _ in range(n_layers)
        ])

        # Feature transformer blocks
        self.feature_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, cross_attention=False)
            for _ in range(n_layers)
        ])

        # Learnable position embeddings
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.feature_pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with alternating attention.

        Args:
            x: Input of shape (batch, n_features, seq_len, d_model)
            context: Optional context for cross-attention (batch, context_len, d_model)

        Returns:
            Output of shape (batch, n_features, seq_len, d_model)
        """
        batch_size = x.shape[0]

        for temp_block, feat_block in zip(self.temporal_blocks, self.feature_blocks):
            # Temporal attention: (batch, n_features, seq_len, d_model) -> (batch * n_features, seq_len, d_model)
            x = rearrange(x, "b f s d -> (b f) s d")
            x = x + self.temporal_pos
            # Reshape context for batched temporal attention
            if context is not None:
                ctx = context.unsqueeze(1).expand(-1, self.n_features, -1, -1)
                ctx = rearrange(ctx, "b f s d -> (b f) s d")
            else:
                ctx = None
            x = temp_block(x, context=ctx)
            x = rearrange(x, "(b f) s d -> b f s d", b=batch_size)

            # Feature attention: (batch, n_features, seq_len, d_model) -> (batch * seq_len, n_features, d_model)
            x = rearrange(x, "b f s d -> (b s) f d")
            x = x + self.feature_pos
            x = feat_block(x)
            x = rearrange(x, "(b s) f d -> b f s d", b=batch_size)

        return x


class TimeSeriesTransformer(nn.Module):
    """Simplified transformer for single-zone forecasting."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, cross_attention=True)
            for _ in range(n_layers)
        ])

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            context: Optional context for cross-attention

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x, context=context)

        return x
