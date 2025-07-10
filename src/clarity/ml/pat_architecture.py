"""PAT Model Architecture - Following SOLID principles.

Defines the Pretrained Actigraphy Transformer architecture.
Based on Dartmouth specifications from arXiv:2411.15240.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models.

    Follows Single Responsibility: Only handles positional encoding.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, : x.size(1)]


class PatchEmbedding(nn.Module):
    """Convert time series to patch embeddings.

    Follows Single Responsibility: Only handles patch embedding.
    """

    def __init__(self, patch_size: int, embed_dim: int):
        """Initialize patch embedding.

        Args:
            patch_size: Size of each patch
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to patch embeddings.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        batch_size, seq_len = x.shape

        # Ensure sequence length is divisible by patch size
        assert (
            seq_len % self.patch_size == 0
        ), f"Sequence length {seq_len} must be divisible by patch size {self.patch_size}"

        # Reshape to patches
        num_patches = seq_len // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size)

        # Project patches to embeddings
        embeddings = self.projection(x)

        return embeddings


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention.

    Implements the non-standard attention mechanism from Dartmouth PAT.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        """Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Multi-head attention (non-standard: key_dim = embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of same shape
        """
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class PATModel(nn.Module):
    """Pretrained Actigraphy Transformer model.

    Implements the architecture from Dartmouth's paper.
    Follows Open/Closed principle: Extensible but stable interface.
    """

    def __init__(self, config: "ModelConfig"):
        """Initialize PAT model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size, embed_dim=config.embed_dim
        )

        # Positional encoding
        max_patches = config.input_size // config.patch_size
        self.pos_encoding = PositionalEncoding(
            embed_dim=config.embed_dim, max_len=max_patches
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ff_dim=config.ff_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Output projection (for embeddings)
        self.output_norm = nn.LayerNorm(config.embed_dim)

        logger.info(
            "Initialized %s model with %d layers, %d heads, %d embed_dim",
            config.name,
            config.num_layers,
            config.num_heads,
            config.embed_dim,
        )

    def forward(self, x: torch.Tensor, return_embeddings: bool = True) -> torch.Tensor:
        """Forward pass through PAT model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            return_embeddings: Return patch embeddings (vs. predictions)

        Returns:
            Output embeddings of shape (batch_size, num_patches, embed_dim)
        """
        # Convert to patches
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.output_norm(x)

        return x

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get patch-level embeddings for input.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Patch embeddings
        """
        return self.forward(x, return_embeddings=True)

    def get_sequence_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence-level embedding (mean pooled).

        Args:
            x: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Sequence embedding of shape (batch_size, embed_dim)
        """
        patch_embeddings = self.forward(x, return_embeddings=True)

        # Mean pool across patches
        sequence_embedding = patch_embeddings.mean(dim=1)

        return sequence_embedding
