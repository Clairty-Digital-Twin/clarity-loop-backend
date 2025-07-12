"""Comprehensive unit tests for PAT Model Loader.

Tests all components of the PAT model loader including:
- ModelSize enum
- ModelVersion dataclass
- ModelCache functionality
- PATModelLoader with all features
- Error handling and edge cases
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
import torch
from torch import nn

from clarity.ml.pat_architecture import ModelConfig, PATModel
from clarity.ml.pat_model_loader import (
    ModelCache,
    ModelLoadError,
    ModelSize,
    ModelVersion,
    PATModelLoader,
    get_model_config,
)
from clarity.services.s3_storage_service import S3StorageService


class TestModelSize:
    """Test ModelSize enum."""

    def test_model_size_values(self) -> None:
        """Test ModelSize enum has expected values."""
        assert ModelSize.SMALL == "small"
        assert ModelSize.MEDIUM == "medium"
        assert ModelSize.LARGE == "large"

    def test_model_size_iteration(self) -> None:
        """Test iterating over ModelSize values."""
        sizes = list(ModelSize)
        assert len(sizes) == 3
        assert ModelSize.SMALL in sizes
        assert ModelSize.MEDIUM in sizes
        assert ModelSize.LARGE in sizes


class TestGetModelConfig:
    """Test get_model_config function."""

    def test_get_model_config_small(self) -> None:
        """Test getting config for small model."""
        config = get_model_config(ModelSize.SMALL)

        assert config.name == "PAT-S"
        assert config.input_size == 10080
        assert config.patch_size == 18
        assert config.embed_dim == 96
        assert config.num_layers == 1
        assert config.num_heads == 6
        assert config.ff_dim == 256
        assert config.dropout == 0.1

    def test_get_model_config_medium(self) -> None:
        """Test getting config for medium model."""
        config = get_model_config(ModelSize.MEDIUM)

        assert config.name == "PAT-M"
        assert config.num_layers == 2
        assert config.num_heads == 12
        # Other params same as small

    def test_get_model_config_large(self) -> None:
        """Test getting config for large model."""
        config = get_model_config(ModelSize.LARGE)

        assert config.name == "PAT-L"
        assert config.patch_size == 9
        assert config.num_layers == 4
        assert config.num_heads == 12

    def test_get_model_config_all_sizes(self) -> None:
        """Test getting configs for all model sizes."""
        for size in ModelSize:
            config = get_model_config(size)
            assert isinstance(config, ModelConfig)
            assert config.name.startswith("PAT-")


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_model_version_creation(self) -> None:
        """Test creating a ModelVersion."""
        version = ModelVersion(
            version="v1.0",
            timestamp=time.time(),
            checksum="abc123",
            size=ModelSize.MEDIUM,
            metrics={"accuracy": 0.95, "loss": 0.05},
        )

        assert version.version == "v1.0"
        assert isinstance(version.timestamp, float)
        assert version.checksum == "abc123"
        assert version.size == ModelSize.MEDIUM
        assert version.metrics["accuracy"] == 0.95


class TestModelCache:
    """Test ModelCache functionality."""

    def test_cache_initialization(self) -> None:
        """Test ModelCache initialization."""
        cache = ModelCache(ttl_seconds=3600)

        assert cache._ttl == 3600
        assert isinstance(cache._cache, dict)
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self) -> None:
        """Test setting and getting from cache."""
        cache = ModelCache(ttl_seconds=3600)
        model = MagicMock()

        cache.set("test_key", model)
        retrieved = cache.get("test_key")

        assert retrieved is model

    def test_cache_get_nonexistent(self) -> None:
        """Test getting non-existent key from cache."""
        cache = ModelCache()

        result = cache.get("nonexistent")
        assert result is None

    def test_cache_ttl_expiration(self) -> None:
        """Test cache TTL expiration."""
        cache = ModelCache(ttl_seconds=0.1)  # 100ms TTL
        model = MagicMock()

        cache.set("test_key", model)
        assert cache.get("test_key") is model

        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("test_key") is None

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        cache = ModelCache()

        cache.set("key1", "model1")
        cache.set("key2", "model2")
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0


class TestPATModelLoader:
    """Test PATModelLoader functionality."""

    @pytest.fixture
    def model_dir(self, tmp_path: Path) -> Path:
        """Create temporary model directory."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        return model_dir

    @pytest.fixture
    def mock_s3_service(self) -> MagicMock:
        """Create mock S3 service."""
        return MagicMock(spec=S3StorageService)

    @pytest.fixture
    def loader(self, model_dir: Path, mock_s3_service: MagicMock) -> PATModelLoader:
        """Create PATModelLoader instance."""
        return PATModelLoader(
            model_dir=model_dir,
            s3_service=mock_s3_service,
            cache_ttl=3600,
            enable_hot_swap=False,
        )

    def test_loader_initialization(self, model_dir: Path) -> None:
        """Test PATModelLoader initialization."""
        with patch("clarity.ml.pat_model_loader.logger") as mock_logger:
            loader = PATModelLoader(
                model_dir=model_dir,
                s3_service=None,
                cache_ttl=1800,
                enable_hot_swap=True,
            )

            assert loader.model_dir == model_dir
            assert loader.s3_service is None
            assert loader.enable_hot_swap is True
            assert isinstance(loader._cache, ModelCache)
            assert loader._cache._ttl == 1800

            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_from_cache(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test loading model from cache."""
        # Create mock model
        mock_model = MagicMock(spec=nn.Module)
        loader._cache.set("medium:latest", mock_model)

        # Load model (should come from cache)
        model = await loader.load_model(ModelSize.MEDIUM)

        assert model is mock_model

    @pytest.mark.asyncio
    async def test_load_model_from_file(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test loading model from file."""
        # Create dummy model file
        model_path = model_dir / "pat_small.pth"

        # Create a real model state dict
        config = get_model_config(ModelSize.SMALL)
        real_model = PATModel(config)
        torch.save(real_model.state_dict(), model_path)

        # Load model
        model = await loader.load_model(ModelSize.SMALL)

        assert isinstance(model, nn.Module)
        assert loader._cache.get("small:latest") is model

    @pytest.mark.asyncio
    async def test_load_model_specific_version(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test loading specific model version."""
        # Create versioned model file
        model_path = model_dir / "pat_medium_v2.pth"

        config = get_model_config(ModelSize.MEDIUM)
        real_model = PATModel(config)
        torch.save(real_model.state_dict(), model_path)

        # Load specific version
        model = await loader.load_model(ModelSize.MEDIUM, version="2")

        assert isinstance(model, nn.Module)

    @pytest.mark.asyncio
    async def test_load_model_from_s3(
        self, loader: PATModelLoader, model_dir: Path, mock_s3_service: MagicMock
    ) -> None:
        """Test loading model from S3 when not found locally."""
        # Setup S3 mock to return model data
        config = get_model_config(ModelSize.LARGE)
        real_model = PATModel(config)
        model_bytes = torch.save(real_model.state_dict(), "buffer")

        # Mock the S3 download
        mock_s3_service.download_file = AsyncMock(return_value=b"fake_model_data")

        # Mock torch.load to return a state dict
        with patch("torch.load") as mock_load:
            mock_load.return_value = real_model.state_dict()

            # Load model (should download from S3)
            model = await loader.load_model(ModelSize.LARGE)

            assert isinstance(model, nn.Module)
            mock_s3_service.download_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_force_reload(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test force reloading model even if cached."""
        # Add model to cache
        cached_model = MagicMock()
        loader._cache.set("small:latest", cached_model)

        # Create real model file
        model_path = model_dir / "pat_small.pth"
        config = get_model_config(ModelSize.SMALL)
        real_model = PATModel(config)
        torch.save(real_model.state_dict(), model_path)

        # Force reload
        model = await loader.load_model(ModelSize.SMALL, force_reload=True)

        assert model is not cached_model
        assert isinstance(model, nn.Module)

    @pytest.mark.asyncio
    async def test_load_model_error_handling(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test error handling when loading model fails."""
        # Remove S3 service to ensure it doesn't try to download
        loader.s3_service = None

        # Try to load non-existent model
        with pytest.raises(
            ModelLoadError, match=r"Failed to load .* model:.*No such file or directory"
        ):
            await loader.load_model(ModelSize.SMALL)

    @pytest.mark.asyncio
    async def test_load_model_invalid_weights(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test error handling for invalid model weights."""
        # Create file with invalid data
        model_path = model_dir / "pat_small.pth"
        model_path.write_bytes(b"invalid model data")

        with pytest.raises(ModelLoadError, match="Failed to load"):
            await loader.load_model(ModelSize.SMALL)

    def test_get_versioned_path(self, loader: PATModelLoader) -> None:
        """Test getting versioned model path."""
        path = loader._get_versioned_path(ModelSize.MEDIUM, "1.2")
        expected = loader.model_dir / "pat_medium_v1.2.pth"
        assert path == expected

    def test_get_latest_path_with_versions(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test getting latest model path when versions exist."""
        # Create multiple version files
        (model_dir / "pat_small_v1.pth").touch()
        (model_dir / "pat_small_v2.pth").touch()
        (model_dir / "pat_small_v3.pth").touch()

        latest_path = loader._get_latest_path(ModelSize.SMALL)
        # With lexicographic sort, v3 comes after v2 and v1
        assert latest_path.name == "pat_small_v3.pth"

    def test_get_latest_path_no_versions(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test getting latest path when no versions exist."""
        latest_path = loader._get_latest_path(ModelSize.LARGE)
        expected = model_dir / "pat_large.pth"
        assert latest_path == expected

    @pytest.mark.asyncio
    async def test_download_from_s3_error(
        self, loader: PATModelLoader, mock_s3_service: MagicMock
    ) -> None:
        """Test error handling in S3 download."""
        mock_s3_service.download_file = AsyncMock(side_effect=Exception("S3 error"))

        with pytest.raises(ModelLoadError, match="Failed to download model from S3"):
            await loader._download_from_s3(
                ModelSize.SMALL, "1.0", Path("/tmp/test.pth")
            )

    def test_validate_model(self, loader: PATModelLoader) -> None:
        """Test model validation."""
        config = get_model_config(ModelSize.SMALL)
        model = PATModel(config)

        # Should not raise
        loader._validate_model(model, config)

    def test_validate_model_wrong_output(self, loader: PATModelLoader) -> None:
        """Test model validation with wrong output shape."""
        config = get_model_config(ModelSize.SMALL)

        # Create mock model with wrong output
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(1, 10, 10)  # Wrong shape

        with pytest.raises(ModelLoadError, match="Invalid output shape"):
            loader._validate_model(mock_model, config)

    def test_update_current_version(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test updating current version tracking."""
        model_path = model_dir / "test_model.pth"
        model_path.write_bytes(b"test data")

        loader._update_current_version(ModelSize.SMALL, "v1.0", model_path)

        assert ModelSize.SMALL in loader._current_versions
        version = loader._current_versions[ModelSize.SMALL]
        assert version.version == "v1.0"
        assert version.size == ModelSize.SMALL
        assert len(version.checksum) > 0

    @pytest.mark.asyncio
    async def test_fallback_to_previous(
        self, loader: PATModelLoader, model_dir: Path
    ) -> None:
        """Test fallback to previous version."""
        # Set current version - remove S3 service to avoid download attempts
        loader.s3_service = None
        loader._current_versions[ModelSize.MEDIUM] = ModelVersion(
            version="v3",
            timestamp=time.time(),
            checksum="abc",
            size=ModelSize.MEDIUM,
            metrics={},
        )

        # Create previous version file
        model_path = model_dir / "pat_medium_v2.pth"
        config = get_model_config(ModelSize.MEDIUM)
        model = PATModel(config)
        torch.save(model.state_dict(), model_path)

        # Test fallback
        with patch("clarity.ml.pat_model_loader.logger") as mock_logger:
            fallback_model = await loader.fallback_to_previous(ModelSize.MEDIUM)

            assert isinstance(fallback_model, nn.Module)
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_no_current_version(self, loader: PATModelLoader) -> None:
        """Test fallback when no current version exists."""
        with pytest.raises(ModelLoadError, match="No current version"):
            await loader.fallback_to_previous(ModelSize.SMALL)

    def test_get_metrics(self, loader: PATModelLoader) -> None:
        """Test getting loader metrics."""
        # Add some load times
        loader._load_times.extend([0.1, 0.2, 0.15])

        # Add cached models
        loader._cache.set("test1", MagicMock())
        loader._cache.set("test2", MagicMock())

        # Add current versions
        loader._current_versions[ModelSize.SMALL] = ModelVersion(
            version="v1",
            timestamp=time.time(),
            checksum="",
            size=ModelSize.SMALL,
            metrics={},
        )

        metrics = loader.get_metrics()

        assert metrics["average_load_time_ms"] == 150.0  # (0.1 + 0.2 + 0.15) / 3 * 1000
        assert metrics["total_loads"] == 3
        assert metrics["cached_models"] == 2
        assert metrics["current_versions"]["small"] == "v1"

    def test_get_metrics_empty(self, loader: PATModelLoader) -> None:
        """Test getting metrics when no data."""
        metrics = loader.get_metrics()
        assert metrics == {}

    def test_clear_cache(self, loader: PATModelLoader) -> None:
        """Test clearing model cache."""
        # Add some models to cache
        loader._cache.set("model1", MagicMock())
        loader._cache.set("model2", MagicMock())

        with patch("clarity.ml.pat_model_loader.logger") as mock_logger:
            loader.clear_cache()

            assert len(loader._cache._cache) == 0
            mock_logger.info.assert_called_with("Model cache cleared")
