"""Comprehensive unit tests for PAT Model Architecture.

Tests all components of the PAT architecture including:
- ModelConfig dataclass
- PositionalEncoding module
- PatchEmbedding module
- TransformerBlock module
- PATModel main architecture
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from clarity.ml.pat_architecture import (
    ModelConfig,
    PatchEmbedding,
    PATModel,
    PositionalEncoding,
    TransformerBlock,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_creation(self) -> None:
        """Test creating a ModelConfig with all parameters."""
        config = ModelConfig(
            name="test-model",
            input_size=10080,
            patch_size=18,
            embed_dim=96,
            num_layers=2,
            num_heads=12,
            ff_dim=256,
            dropout=0.1,
        )

        assert config.name == "test-model"
        assert config.input_size == 10080
        assert config.patch_size == 18
        assert config.embed_dim == 96
        assert config.num_layers == 2
        assert config.num_heads == 12
        assert config.ff_dim == 256
        assert config.dropout == 0.1

    def test_model_config_default_dropout(self) -> None:
        """Test ModelConfig uses default dropout value."""
        config = ModelConfig(
            name="test",
            input_size=1000,
            patch_size=10,
            embed_dim=64,
            num_layers=1,
            num_heads=8,
            ff_dim=128,
        )

        assert config.dropout == 0.1


class TestPositionalEncoding:
    """Test PositionalEncoding module."""

    def test_positional_encoding_initialization(self) -> None:
        """Test PositionalEncoding initialization."""
        embed_dim = 64
        max_len = 100
        pe = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        # Check the registered buffer
        assert hasattr(pe, "pe")
        assert pe.pe.shape == (1, max_len, embed_dim)

    def test_positional_encoding_forward(self) -> None:
        """Test PositionalEncoding forward pass."""
        embed_dim = 64
        seq_len = 50
        batch_size = 8

        pe = PositionalEncoding(embed_dim=embed_dim)
        x = torch.randn(batch_size, seq_len, embed_dim)

        output = pe(x)

        # Output should have same shape as input
        assert output.shape == x.shape

        # Output should be different from input (positional encoding added)
        assert not torch.allclose(output, x)

    def test_positional_encoding_deterministic(self) -> None:
        """Test that positional encoding is deterministic."""
        embed_dim = 32
        pe = PositionalEncoding(embed_dim=embed_dim)

        x = torch.zeros(2, 10, embed_dim)
        output1 = pe(x)
        output2 = pe(x)

        # Same input should produce same output
        assert torch.allclose(output1, output2)

    def test_positional_encoding_values(self) -> None:
        """Test that positional encoding values are reasonable."""
        embed_dim = 64
        pe = PositionalEncoding(embed_dim=embed_dim)

        # The positional encoding for position 0 should be all zeros or small values
        pos_0 = pe.pe[0, 0, :]
        assert pos_0.abs().max() < 2.0  # Reasonable bound for sin/cos values


class TestPatchEmbedding:
    """Test PatchEmbedding module."""

    def test_patch_embedding_initialization(self) -> None:
        """Test PatchEmbedding initialization."""
        patch_size = 18
        embed_dim = 96

        patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)

        assert patch_embed.patch_size == patch_size
        assert hasattr(patch_embed, "projection")
        assert isinstance(patch_embed.projection, nn.Linear)
        assert patch_embed.projection.in_features == patch_size
        assert patch_embed.projection.out_features == embed_dim

    def test_patch_embedding_forward_valid(self) -> None:
        """Test PatchEmbedding forward pass with valid input."""
        patch_size = 10
        embed_dim = 64
        batch_size = 4
        seq_len = 100  # Divisible by patch_size

        patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        x = torch.randn(batch_size, seq_len)

        output = patch_embed(x)

        expected_num_patches = seq_len // patch_size
        assert output.shape == (batch_size, expected_num_patches, embed_dim)

    def test_patch_embedding_forward_invalid_length(self) -> None:
        """Test PatchEmbedding raises error for invalid sequence length."""
        patch_size = 10
        embed_dim = 64
        seq_len = 95  # Not divisible by patch_size

        patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        x = torch.randn(2, seq_len)

        with pytest.raises(ValueError, match="must be divisible by patch size"):
            patch_embed(x)

    def test_patch_embedding_different_batch_sizes(self) -> None:
        """Test PatchEmbedding with different batch sizes."""
        patch_size = 18
        embed_dim = 96
        seq_len = 180

        patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)

        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, seq_len)
            output = patch_embed(x)
            assert output.shape == (batch_size, seq_len // patch_size, embed_dim)


class TestTransformerBlock:
    """Test TransformerBlock module."""

    def test_transformer_block_initialization(self) -> None:
        """Test TransformerBlock initialization."""
        embed_dim = 96
        num_heads = 12
        ff_dim = 256
        dropout = 0.1

        block = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Check all components are initialized
        assert hasattr(block, "attention")
        assert isinstance(block.attention, nn.MultiheadAttention)
        assert hasattr(block, "ff")
        assert isinstance(block.ff, nn.Sequential)
        assert hasattr(block, "norm1")
        assert isinstance(block.norm1, nn.LayerNorm)
        assert hasattr(block, "norm2")
        assert isinstance(block.norm2, nn.LayerNorm)
        assert hasattr(block, "dropout")
        assert isinstance(block.dropout, nn.Dropout)

    def test_transformer_block_forward(self) -> None:
        """Test TransformerBlock forward pass."""
        embed_dim = 64
        num_heads = 8
        ff_dim = 128
        batch_size = 4
        seq_len = 10

        block = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )

        x = torch.randn(batch_size, seq_len, embed_dim)
        output = block(x)

        # Output shape should match input shape
        assert output.shape == x.shape

    def test_transformer_block_residual_connections(self) -> None:
        """Test that transformer block uses residual connections."""
        embed_dim = 64
        num_heads = 8
        ff_dim = 128

        block = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=0.0,  # No dropout for testing
        )

        # Set to eval mode to disable dropout
        block.eval()

        # Create input that should produce different outputs
        x = torch.randn(2, 5, embed_dim)

        with torch.no_grad():
            output = block(x)

        # Due to residual connections, output should contain information from input
        # but should not be identical (due to transformations)
        assert not torch.allclose(output, x, atol=1e-5)


class TestPATModel:
    """Test PATModel main architecture."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create a test model configuration."""
        return ModelConfig(
            name="test-pat",
            input_size=180,  # Small for testing
            patch_size=18,
            embed_dim=64,
            num_layers=2,
            num_heads=8,
            ff_dim=128,
            dropout=0.1,
        )

    def test_pat_model_initialization(self, model_config: ModelConfig) -> None:
        """Test PATModel initialization."""
        with patch("clarity.ml.pat_architecture.logger") as mock_logger:
            model = PATModel(model_config)

            # Check all components are initialized
            assert hasattr(model, "patch_embed")
            assert isinstance(model.patch_embed, PatchEmbedding)
            assert hasattr(model, "pos_encoding")
            assert isinstance(model.pos_encoding, PositionalEncoding)
            assert hasattr(model, "transformer_blocks")
            assert isinstance(model.transformer_blocks, nn.ModuleList)
            assert len(model.transformer_blocks) == model_config.num_layers
            assert hasattr(model, "output_norm")
            assert isinstance(model.output_norm, nn.LayerNorm)

            # Check logger was called
            mock_logger.info.assert_called_once()

    def test_pat_model_forward(self, model_config: ModelConfig) -> None:
        """Test PATModel forward pass."""
        model = PATModel(model_config)
        batch_size = 4

        x = torch.randn(batch_size, model_config.input_size)
        output = model(x)

        expected_num_patches = model_config.input_size // model_config.patch_size
        assert output.shape == (
            batch_size,
            expected_num_patches,
            model_config.embed_dim,
        )

    def test_pat_model_get_patch_embeddings(self, model_config: ModelConfig) -> None:
        """Test getting patch embeddings."""
        model = PATModel(model_config)
        batch_size = 2

        x = torch.randn(batch_size, model_config.input_size)
        embeddings = model.get_patch_embeddings(x)

        expected_num_patches = model_config.input_size // model_config.patch_size
        assert embeddings.shape == (
            batch_size,
            expected_num_patches,
            model_config.embed_dim,
        )

    def test_pat_model_get_sequence_embedding(self, model_config: ModelConfig) -> None:
        """Test getting sequence-level embedding."""
        model = PATModel(model_config)
        batch_size = 3

        x = torch.randn(batch_size, model_config.input_size)
        seq_embedding = model.get_sequence_embedding(x)

        # Sequence embedding should be mean-pooled
        assert seq_embedding.shape == (batch_size, model_config.embed_dim)

    def test_pat_model_eval_mode(self, model_config: ModelConfig) -> None:
        """Test model behavior in eval mode."""
        model = PATModel(model_config)
        model.eval()

        x = torch.randn(1, model_config.input_size)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        # In eval mode, same input should produce same output
        assert torch.allclose(output1, output2)

    def test_pat_model_different_configs(self) -> None:
        """Test PATModel with different configurations."""
        configs = [
            ModelConfig("small", 180, 18, 32, 1, 4, 64),
            ModelConfig("medium", 360, 36, 64, 2, 8, 128),
            ModelConfig("large", 720, 18, 96, 4, 12, 256),
        ]

        for config in configs:
            model = PATModel(config)
            x = torch.randn(1, config.input_size)
            output = model(x)

            expected_patches = config.input_size // config.patch_size
            assert output.shape == (1, expected_patches, config.embed_dim)

    def test_pat_model_gradient_flow(self, model_config: ModelConfig) -> None:
        """Test that gradients flow through the model."""
        model = PATModel(model_config)
        x = torch.randn(2, model_config.input_size, requires_grad=True)

        output = model(x)
        loss = output.mean()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        # Check gradients are not exactly zero (allow for very small values)
        assert x.grad.abs().max() > 0

        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
