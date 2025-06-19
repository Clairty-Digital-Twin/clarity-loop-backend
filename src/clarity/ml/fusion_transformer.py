"""FusionTransformer - Multimodal Health Data Fusion.

Combines multiple health modality features into a unified health state vector
using transformer-based attention mechanisms for cross-modal interactions.
"""

# removed - breaks FastAPI

import logging

from pydantic import BaseModel, Field
import torch
from torch import nn

logger = logging.getLogger(__name__)


class FusionConfig(BaseModel):
    """Configuration for FusionTransformer."""

    modality_dims: dict[str, int] = Field(description="Dimensions for each modality")
    embed_dim: int = Field(default=64, description="Common embedding dimension")
    num_heads: int = Field(default=4, description="Number of attention heads")
    num_layers: int = Field(default=2, description="Number of transformer layers")
    dropout: float = Field(default=0.1, description="Dropout rate")
    output_dim: int = Field(default=64, description="Final output dimension")


class FusionTransformer(nn.Module):
    """Multimodal fusion transformer for health data.

    Uses transformer encoder with CLS token to fuse multiple health modalities
    (cardio, respiratory, activity, etc.) into a unified health state vector.
    """

    def __init__(self, config: FusionConfig) -> None:
        """Initialize FusionTransformer.

        Args:
            config: Fusion configuration with modality dimensions and model params
        """
        super().__init__()

        self.config = config
        self.modality_names = list(config.modality_dims.keys())

        # Linear projection for each modality to common embedding dimension
        self.modality_projections = nn.ModuleDict(
            {
                name: nn.Linear(dim, config.embed_dim)
                for name, dim in config.modality_dims.items()
            }
        )

        # Learnable [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Positional embeddings for modalities (optional but can help)
        self.modality_embeddings = nn.Embedding(
            len(config.modality_dims) + 1,
            config.embed_dim,  # +1 for CLS token
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,  # Use batch_first=True for easier handling
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Output projection to final dimension
        self.output_projection = nn.Linear(config.embed_dim, config.output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)

        # Initialize projection layers
        for projection in self.modality_projections.values():
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through fusion transformer.

        Args:
            inputs: Dictionary mapping modality names to feature tensors
                   Each tensor should have shape (batch_size, feature_dim)

        Returns:
            Unified health state vector of shape (batch_size, output_dim)
        """
        batch_size = None
        modality_tokens = []

        # Process each modality
        for i, (modality_name, features) in enumerate(inputs.items()):
            if modality_name not in self.modality_projections:
                logger.warning("Unknown modality: %s, skipping", modality_name)
                continue

            if batch_size is None:
                batch_size = features.size(0)

            # Project modality features to common embedding space
            projected = self.modality_projections[modality_name](
                features
            )  # (batch, embed_dim)

            # Add positional embedding for this modality
            pos_id = torch.tensor(
                i + 1, device=features.device
            )  # +1 because 0 is for CLS
            pos_embedding = self.modality_embeddings(pos_id)  # (embed_dim,)
            projected += pos_embedding.unsqueeze(0)  # Broadcast to (batch, embed_dim)

            # Add sequence dimension: (batch, 1, embed_dim)
            modality_tokens.append(projected.unsqueeze(1))

        if not modality_tokens:
            # No valid modalities, return zero vector
            return torch.zeros(batch_size or 1, self.config.output_dim)

        # Concatenate all modality tokens
        modality_sequence = torch.cat(
            modality_tokens, dim=1
        )  # (batch, num_modalities, embed_dim)

        # Prepare CLS token
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # type: ignore[arg-type]  # (batch, 1, embed_dim)

        # Add CLS positional embedding
        cls_pos_embedding = self.modality_embeddings(
            torch.tensor(0, device=cls_tokens.device)
        )
        cls_tokens += cls_pos_embedding.unsqueeze(0).unsqueeze(0)

        # Concatenate CLS token with modality tokens
        sequence = torch.cat(
            [cls_tokens, modality_sequence], dim=1
        )  # (batch, 1+num_modalities, embed_dim)

        # Apply transformer encoder
        encoded = self.transformer(sequence)  # (batch, 1+num_modalities, embed_dim)

        # Extract CLS token output (first token)
        cls_output = encoded[:, 0, :]  # (batch, embed_dim)

        # Project to final output dimension
        output = self.output_projection(cls_output)  # (batch, output_dim)

        # Apply layer normalization
        return self.layer_norm(output)  # type: ignore[no-any-return]

    def get_attention_weights(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get attention weights for interpretability.

        Args:
            inputs: Dictionary mapping modality names to feature tensors

        Returns:
            Attention weights tensor
        """
        # This would require modifying the transformer to return attention weights
        # For now, return a placeholder
        batch_size = next(iter(inputs.values())).size(0)
        num_modalities = len(inputs)
        return torch.ones(
            batch_size, self.config.num_heads, num_modalities + 1, num_modalities + 1
        )


class HealthFusionService:
    """Service for managing health data fusion."""

    def __init__(self, device: str = "cpu") -> None:
        """Initialize fusion service.

        Args:
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model: FusionTransformer | None = None
        self.config: FusionConfig | None = None
        self.logger = logging.getLogger(__name__)

    def initialize_model(self, modality_dims: dict[str, int]) -> None:
        """Initialize fusion model with given modality dimensions.

        Args:
            modality_dims: Dictionary mapping modality names to feature dimensions
        """
        self.config = FusionConfig(
            modality_dims=modality_dims,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            output_dim=64,
        )

        self.model = FusionTransformer(self.config)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        self.logger.info(
            "Initialized fusion model with modalities: %s", list(modality_dims.keys())
        )

    def fuse_modalities(self, modality_features: dict[str, list[float]]) -> list[float]:
        """Fuse multiple modality features into unified health vector.

        Args:
            modality_features: Dictionary mapping modality names to feature lists

        Returns:
            Unified health state vector as list of floats
        """
        if self.model is None:
            msg = "Model not initialized. Call initialize_model() first."
            raise RuntimeError(msg)

        try:
            # Convert to tensors
            tensor_inputs = {}
            for modality, features in modality_features.items():
                if modality in self.config.modality_dims:  # type: ignore[union-attr]
                    tensor_inputs[modality] = torch.tensor(
                        [features], dtype=torch.float32, device=self.device
                    )  # Add batch dimension

            if not tensor_inputs:
                self.logger.warning("No valid modalities provided for fusion")
                return [0.0] * self.config.output_dim  # type: ignore[union-attr]

            # Run fusion
            with torch.no_grad():
                fused_vector = self.model(tensor_inputs)

            # Convert back to list
            result = fused_vector.cpu().numpy().tolist()[0]  # Remove batch dimension

        except Exception:
            self.logger.exception("Error during fusion")
            # Return zero vector on error
            return [0.0] * (self.config.output_dim if self.config else 64)
        else:
            self.logger.info(
                "Fused %d modalities into %d-dim vector",
                len(tensor_inputs),
                len(result),
            )

            return result  # type: ignore[no-any-return]


class FusionServiceSingleton:
    """Singleton container for fusion service."""

    _instance: HealthFusionService | None = None

    @classmethod
    def get_instance(cls) -> HealthFusionService:
        """Get or create fusion service instance."""
        if cls._instance is None:
            cls._instance = HealthFusionService()
        return cls._instance


def get_fusion_service() -> HealthFusionService:
    """Get or create global fusion service instance."""
    return FusionServiceSingleton.get_instance()
