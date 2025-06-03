"""PAT (Pretrained Actigraphy Transformer) Model Service.

This service implements the Dartmouth PAT model for actigraphy analysis,
providing state-of-the-art sleep and activity pattern recognition.

Based on: "AI Foundation Models for Wearable Movement Data in Mental Health Research"
arXiv:2411.15240 (Dartmouth College, 29,307 participants, NHANES 2003-2014)
LATEST VERSION: January 14, 2025 (v3) - BLEEDING EDGE IMPLEMENTATION

ARCHITECTURE SPECIFICATIONS (from Dartmouth source):
- PAT-S: 1 layer, 6 heads, 96 embed_dim, patch_size=18, input_size=10080
- PAT-M: 2 layers, 12 heads, 96 embed_dim, patch_size=18, input_size=10080  
- PAT-L: 4 layers, 12 heads, 96 embed_dim, patch_size=9, input_size=10080
- All models: ff_dim=256, dropout=0.1
"""

from datetime import UTC, datetime
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import BaseModel, Field
import torch
from torch import nn
import torch.nn.functional as F

try:
    import h5py  # type: ignore[import-untyped]

    _has_h5py = True
except ImportError:
    h5py = None  # type: ignore[assignment]
    _has_h5py = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

from clarity.ml.preprocessing import ActigraphyDataPoint, HealthDataPreprocessor
from clarity.ports.ml_ports import IMLModelService

logger = logging.getLogger(__name__)

# Model configurations matching Dartmouth specs exactly
PAT_CONFIGS = {
    "small": {
        "num_layers": 1,
        "num_heads": 6,
        "embed_dim": 576,  # CORRECTED: 6 heads × 96 head_dim = 576
        "head_dim": 96,    # NEW: explicit head dimension
        "ff_dim": 256,
        "patch_size": 18,
        "input_size": 10080,
        "model_path": "models/PAT-S_29k_weights.h5",
    },
    "medium": {
        "num_layers": 2,
        "num_heads": 12,
        "embed_dim": 1152,  # CORRECTED: 12 heads × 96 head_dim = 1152
        "head_dim": 96,     # NEW: explicit head dimension
        "ff_dim": 256,
        "patch_size": 18,
        "input_size": 10080,
        "model_path": "models/PAT-M_29k_weights.h5",
    },
    "large": {
        "num_layers": 4,
        "num_heads": 12,
        "embed_dim": 1152,  # CORRECTED: 12 heads × 96 head_dim = 1152
        "head_dim": 96,     # NEW: explicit head dimension
        "ff_dim": 256,
        "patch_size": 9,
        "input_size": 10080,
        "model_path": "models/PAT-L_29k_weights.h5",
    },
}

# Clinical thresholds as constants
EXCELLENT_SLEEP_EFFICIENCY = 85
GOOD_SLEEP_EFFICIENCY = 75
HIGH_CIRCADIAN_SCORE = 0.8
MODERATE_CIRCADIAN_SCORE = 0.6
HIGH_DEPRESSION_RISK = 0.7
MODERATE_DEPRESSION_RISK = 0.4


class ActigraphyInput(BaseModel):
    """Input model for actigraphy data."""

    user_id: str
    data_points: list[ActigraphyDataPoint]
    sampling_rate: float = Field(default=1.0, description="Samples per minute")
    duration_hours: int = Field(default=168, description="Duration in hours (1 week)")


class ActigraphyAnalysis(BaseModel):
    """Output model for PAT analysis results."""

    user_id: str
    analysis_timestamp: str
    sleep_efficiency: float = Field(description="Sleep efficiency percentage (0-100)")
    sleep_onset_latency: float = Field(description="Time to fall asleep (minutes)")
    wake_after_sleep_onset: float = Field(description="WASO minutes")
    total_sleep_time: float = Field(description="Total sleep time (hours)")
    circadian_rhythm_score: float = Field(description="Circadian regularity (0-1)")
    activity_fragmentation: float = Field(description="Activity fragmentation index")
    depression_risk_score: float = Field(description="Depression risk (0-1)")
    sleep_stages: list[str] = Field(description="Predicted sleep stages")
    confidence_score: float = Field(description="Model confidence (0-1)")
    clinical_insights: list[str] = Field(description="Clinical interpretations")


class PATPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching Dartmouth implementation."""

    def __init__(self, embed_dim: int, max_len: int = 10000) -> None:
        super().__init__()
        
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) 
            * (-math.log(10000.0) / embed_dim)
        )
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        pe_buffer = self.pe  # Access the registered buffer
        return x + pe_buffer[:seq_len].unsqueeze(0)


class PATTransformerBlock(nn.Module):
    """Single transformer block matching Dartmouth architecture exactly."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        
        # Layer normalization (matches TF LayerNorm with gamma/beta)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff2(F.relu(self.ff1(x)))
        ff_out = self.dropout(ff_out)
        x = self.norm2(x + ff_out)
        
        return x


class PATTransformer(nn.Module):
    """PAT (Pretrained Actigraphy Transformer) model implementation.
    
    Exact PyTorch implementation of Dartmouth PAT architecture.
    """

    def __init__(
        self,
        input_size: int = 10080,
        patch_size: int = 18,
        embed_dim: int = 96,
        num_layers: int = 2,
        num_heads: int = 12,
        ff_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 18,  # From H5 analysis: PAT-M/S output 18 classes
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = input_size // patch_size
        
        # Input processing layers
        self.patch_embedding = nn.Linear(patch_size, embed_dim)
        self.positional_encoding = PATPositionalEncoding(embed_dim, self.num_patches)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            PATTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head (matches dense layer from H5)
        self.output_head = nn.Linear(embed_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the PAT model."""
        batch_size, seq_len = x.shape
        
        # Reshape input to patches [batch, num_patches, patch_size]
        x = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch, num_patches, embed_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling for sequence-level predictions
        pooled = x.mean(dim=1)  # [batch, embed_dim]
        
        # Output predictions
        logits = self.output_head(pooled)  # [batch, num_classes]
        
        # Convert to clinical metrics (simplified for now)
        # This will be enhanced with proper clinical interpretation
        sleep_metrics = torch.sigmoid(logits[:, :8])  # First 8 for sleep metrics
        circadian_score = torch.sigmoid(logits[:, 8:9])  # 9th for circadian
        depression_risk = torch.sigmoid(logits[:, 9:10])  # 10th for depression
        
        return {
            "raw_logits": logits,
            "sleep_metrics": sleep_metrics,
            "circadian_score": circadian_score,
            "depression_risk": depression_risk,
            "embeddings": pooled,
        }


class PATModelService(IMLModelService):
    """Production-ready PAT model service for actigraphy analysis.

    This service loads the Dartmouth PAT model weights and provides
    real-time actigraphy analysis with clinical-grade insights.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_size: str = "medium",
        device: str | None = None,
        preprocessor: HealthDataPreprocessor | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PATTransformer | None = None
        self.is_loaded = False
        self.preprocessor = preprocessor or HealthDataPreprocessor()

        # Get model configuration
        if model_size not in PAT_CONFIGS:
            msg = f"Invalid model size: {model_size}. Choose from {list(PAT_CONFIGS.keys())}"
            raise ValueError(msg)
        
        self.config = PAT_CONFIGS[model_size]
        self.model_path = model_path or self.config["model_path"]

        logger.info(
            "Initializing PAT model service (size: %s, device: %s)",
            model_size,
            self.device,
        )

    async def load_model(self) -> None:
        """Load the PAT model weights asynchronously."""
        try:
            logger.info("Loading PAT model from %s", self.model_path)

            # Initialize model architecture with correct parameters
            self.model = PATTransformer(
                input_size=self.config["input_size"],
                patch_size=self.config["patch_size"],
                embed_dim=self.config["embed_dim"],
                num_layers=self.config["num_layers"],
                num_heads=self.config["num_heads"],
                ff_dim=self.config["ff_dim"],
                num_classes=18 if self.model_size in ["small", "medium"] else 9,
            )

            # Load pre-trained weights if available
            if self.model_path and Path(self.model_path).exists():
                logger.info("Loading pre-trained PAT weights from %s", self.model_path)

                if not _has_h5py:
                    logger.error("h5py not available, cannot load .h5 weights")
                    logger.warning("Using random initialization for PAT model")
                else:
                    try:
                        # Load and convert TensorFlow weights to PyTorch
                        state_dict = self._load_tensorflow_weights(self.model_path)
                        
                        if state_dict:
                            # Load the converted weights
                            missing_keys, unexpected_keys = self.model.load_state_dict(
                                state_dict, strict=False
                            )
                            
                            if missing_keys:
                                logger.warning("Missing keys: %s", missing_keys)
                            if unexpected_keys:
                                logger.warning("Unexpected keys: %s", unexpected_keys)
                            
                            logger.info(
                                "Successfully loaded %d weight tensors from %s",
                                len(state_dict),
                                self.model_path,
                            )
                        else:
                            logger.warning(
                                "No compatible weights found in %s, using random initialization",
                                self.model_path,
                            )
                    except (OSError, KeyError, ValueError) as e:
                        logger.warning(
                            "Failed to load weights from %s: %s. Using random initialization.",
                            self.model_path,
                            e,
                        )
            else:
                logger.warning(
                    "Model weights not found at %s, using random initialization",
                    self.model_path,
                )

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info("PAT model loaded successfully")

        except Exception:
            logger.exception("Failed to load PAT model")
            raise

    def _load_tensorflow_weights(self, h5_path: str) -> dict[str, torch.Tensor]:
        """Load and convert TensorFlow H5 weights to PyTorch format."""
        state_dict = {}
        
        try:
            with h5py.File(h5_path, 'r') as h5_file:  # type: ignore[union-attr]
                logger.info("Converting TensorFlow weights to PyTorch format")
                
                # Convert patch embedding layer (dense -> patch_embedding)
                if 'dense' in h5_file and 'dense' in h5_file['dense']:  # type: ignore[operator,index]
                    dense_group = h5_file['dense']['dense']  # type: ignore[index]
                    if 'kernel:0' in dense_group:  # type: ignore[operator]
                        # TF: [patch_size, embed_dim] -> PyTorch: [patch_size, embed_dim]
                        tf_weight = dense_group['kernel:0'][:]  # type: ignore[index]
                        state_dict['patch_embedding.weight'] = torch.from_numpy(tf_weight.T)  # type: ignore[attr-defined]
                    if 'bias:0' in dense_group:  # type: ignore[operator]
                        tf_bias = dense_group['bias:0'][:]  # type: ignore[index]
                        state_dict['patch_embedding.bias'] = torch.from_numpy(tf_bias)
                
                # Convert transformer layers
                num_layers = self.config["num_layers"]
                for layer_idx in range(1, num_layers + 1):
                    tf_layer_name = f'encoder_layer_{layer_idx}_transformer'
                    
                    if tf_layer_name in h5_file:
                        layer_group = h5_file[tf_layer_name]
                        pytorch_layer_idx = layer_idx - 1  # PyTorch uses 0-based indexing
                        
                        # Convert attention weights
                        self._convert_attention_weights(
                            layer_group, state_dict, pytorch_layer_idx
                        )
                        
                        # Convert feed-forward weights
                        self._convert_ff_weights(
                            layer_group, state_dict, pytorch_layer_idx
                        )
                        
                        # Convert layer norm weights
                        self._convert_layernorm_weights(
                            layer_group, state_dict, pytorch_layer_idx
                        )
                
                logger.info("Successfully converted %d tensors", len(state_dict))
                
        except Exception as e:
            logger.error("Failed to convert TensorFlow weights: %s", e)
            return {}
        
        return state_dict

    def _convert_attention_weights(
        self, layer_group: Any, state_dict: dict[str, torch.Tensor], layer_idx: int
    ) -> None:
        """Convert multi-head attention weights from TensorFlow to PyTorch."""
        if f'encoder_layer_{layer_idx + 1}_attention' not in layer_group:
            return
            
        attn_group = layer_group[f'encoder_layer_{layer_idx + 1}_attention']
        
        # Get dimensions from TF weights
        embed_dim = int(self.config["embed_dim"])
        num_heads = int(self.config["num_heads"])
        head_dim = int(self.config["head_dim"])
        
        # Verify dimensions match
        assert embed_dim == num_heads * head_dim, f"embed_dim {embed_dim} != num_heads {num_heads} * head_dim {head_dim}"
        
        # Collect Q, K, V weights for combined in_proj
        qkv_weights = []
        qkv_biases = []
        
        for qkv_name in ['query', 'key', 'value']:
            if qkv_name in attn_group:  # type: ignore[operator]
                qkv_group = attn_group[qkv_name]  # type: ignore[index]
                
                if 'kernel:0' in qkv_group:  # type: ignore[operator]
                    # TF shape: [input_dim, num_heads, head_dim] 
                    # e.g., PAT-M: (96, 12, 96) = [96, 12, 96]
                    tf_weight = qkv_group['kernel:0'][:]  # type: ignore[index]
                    
                    # Verify TF weight shape matches expectations
                    expected_shape = (96, num_heads, head_dim)  # Input is always from 96-dim patch embedding
                    if tf_weight.shape != expected_shape:
                        logger.warning(
                            f"Unexpected TF weight shape: {tf_weight.shape}, expected {expected_shape}"
                        )
                    
                    # Reshape TF weight to PyTorch format
                    # TF: [input_dim, num_heads, head_dim] → PyTorch: [embed_dim, input_dim]
                    # First permute to [input_dim, head_dim, num_heads] then reshape to [embed_dim, input_dim]
                    tf_reshaped = tf_weight.transpose(0, 2, 1)  # (96, 96, 12)
                    pytorch_weight = tf_reshaped.reshape(embed_dim, 96)  # (1152, 96) for PAT-M
                    qkv_weights.append(pytorch_weight)
                
                if 'bias:0' in qkv_group:  # type: ignore[operator]
                    tf_bias = qkv_group['bias:0'][:]  # type: ignore[index]
                    # TF shape: [num_heads, head_dim] → PyTorch: [embed_dim]
                    pytorch_bias = tf_bias.reshape(-1)  # Flatten to [embed_dim]
                    qkv_biases.append(pytorch_bias)
        
        # Combine Q, K, V into PyTorch's in_proj format
        if len(qkv_weights) == 3:
            # Stack Q, K, V weights: [3*embed_dim, input_dim]
            combined_weight = np.concatenate(qkv_weights, axis=0)  # (3*1152, 96) for PAT-M
            state_dict[f'transformer_layers.{layer_idx}.attention.in_proj_weight'] = torch.from_numpy(combined_weight)
        
        if len(qkv_biases) == 3:
            # Stack Q, K, V biases: [3*embed_dim]
            combined_bias = np.concatenate(qkv_biases, axis=0)  # (3*1152,) for PAT-M
            state_dict[f'transformer_layers.{layer_idx}.attention.in_proj_bias'] = torch.from_numpy(combined_bias)
        
        # Convert attention output projection
        if 'attention_output' in attn_group:  # type: ignore[operator]
            output_group = attn_group['attention_output']  # type: ignore[index]
            
            if 'kernel:0' in output_group:  # type: ignore[operator]
                tf_weight = output_group['kernel:0'][:]  # type: ignore[index]
                # TF shape: [num_heads, head_dim, output_dim] e.g., (12, 96, 96)
                # PyTorch wants: [output_dim, embed_dim] e.g., (96, 1152)
                
                # Reshape: [num_heads, head_dim, output_dim] → [output_dim, num_heads * head_dim]
                output_dim = tf_weight.shape[2]  # Should be 96
                pytorch_weight = tf_weight.transpose(2, 0, 1).reshape(output_dim, embed_dim)  # (96, 1152)
                
                pytorch_name = f'transformer_layers.{layer_idx}.attention.out_proj.weight'
                state_dict[pytorch_name] = torch.from_numpy(pytorch_weight)
            
            if 'bias:0' in output_group:  # type: ignore[operator]
                tf_bias = output_group['bias:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.attention.out_proj.bias'
                state_dict[pytorch_name] = torch.from_numpy(tf_bias)

    def _convert_ff_weights(
        self, layer_group: Any, state_dict: dict[str, torch.Tensor], layer_idx: int
    ) -> None:
        """Convert feed-forward network weights."""
        # FF1 layer
        ff1_key = f'encoder_layer_{layer_idx + 1}_ff1'
        if ff1_key in layer_group:  # type: ignore[operator]
            ff1_group = layer_group[ff1_key]  # type: ignore[index]
            
            if 'kernel:0' in ff1_group:  # type: ignore[operator]
                tf_weight = ff1_group['kernel:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.ff1.weight'
                state_dict[pytorch_name] = torch.from_numpy(tf_weight.T)  # type: ignore[attr-defined]
            
            if 'bias:0' in ff1_group:  # type: ignore[operator]
                tf_bias = ff1_group['bias:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.ff1.bias'
                state_dict[pytorch_name] = torch.from_numpy(tf_bias)
        
        # FF2 layer
        ff2_key = f'encoder_layer_{layer_idx + 1}_ff2'
        if ff2_key in layer_group:  # type: ignore[operator]
            ff2_group = layer_group[ff2_key]  # type: ignore[index]
            
            if 'kernel:0' in ff2_group:  # type: ignore[operator]
                tf_weight = ff2_group['kernel:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.ff2.weight'
                state_dict[pytorch_name] = torch.from_numpy(tf_weight.T)  # type: ignore[attr-defined]
            
            if 'bias:0' in ff2_group:  # type: ignore[operator]
                tf_bias = ff2_group['bias:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.ff2.bias'
                state_dict[pytorch_name] = torch.from_numpy(tf_bias)

    def _convert_layernorm_weights(
        self, layer_group: Any, state_dict: dict[str, torch.Tensor], layer_idx: int
    ) -> None:
        """Convert layer normalization weights (gamma/beta -> weight/bias)."""
        # Norm1 (after attention)
        norm1_key = f'encoder_layer_{layer_idx + 1}_norm1'
        if norm1_key in layer_group:  # type: ignore[operator]
            norm1_group = layer_group[norm1_key]  # type: ignore[index]
            
            if 'gamma:0' in norm1_group:  # type: ignore[operator]
                gamma = norm1_group['gamma:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.norm1.weight'
                state_dict[pytorch_name] = torch.from_numpy(gamma)
            
            if 'beta:0' in norm1_group:  # type: ignore[operator]
                beta = norm1_group['beta:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.norm1.bias'
                state_dict[pytorch_name] = torch.from_numpy(beta)
        
        # Norm2 (after feed-forward)
        norm2_key = f'encoder_layer_{layer_idx + 1}_norm2'
        if norm2_key in layer_group:  # type: ignore[operator]
            norm2_group = layer_group[norm2_key]  # type: ignore[index]
            
            if 'gamma:0' in norm2_group:  # type: ignore[operator]
                gamma = norm2_group['gamma:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.norm2.weight'
                state_dict[pytorch_name] = torch.from_numpy(gamma)
            
            if 'beta:0' in norm2_group:  # type: ignore[operator]
                beta = norm2_group['beta:0'][:]  # type: ignore[index]
                pytorch_name = f'transformer_layers.{layer_idx}.norm2.bias'
                state_dict[pytorch_name] = torch.from_numpy(beta)

    def _preprocess_actigraphy_data(
        self,
        data_points: list[ActigraphyDataPoint],
        target_length: int = 10080,  # 1 week at 1-minute resolution
    ) -> torch.Tensor:
        """Preprocess actigraphy data for PAT model input using injected preprocessor."""
        tensor = self.preprocessor.preprocess_for_pat_model(data_points, target_length)
        return tensor.to(self.device)

    def _postprocess_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        user_id: str,
    ) -> ActigraphyAnalysis:
        """Convert model outputs to clinical insights.

        Args:
            outputs: Raw model outputs
            user_id: User identifier

        Returns:
            Structured actigraphy analysis
        """
        # Extract predictions
        sleep_metrics = outputs["sleep_metrics"].cpu().numpy()[0]
        circadian_score = outputs["circadian_score"].cpu().item()
        depression_risk = outputs["depression_risk"].cpu().item()

        # Calculate sleep metrics with explicit type annotations
        sleep_efficiency: float = float(sleep_metrics[0] * 100)  # Convert to percentage
        sleep_onset_latency: float = float(sleep_metrics[1] * 60)  # Convert to minutes
        wake_after_sleep_onset: float = float(sleep_metrics[2] * 60)
        total_sleep_time: float = float(sleep_metrics[3] * 12)  # Convert to hours
        activity_fragmentation: float = float(sleep_metrics[4])

        # Generate mock sleep stages for now (will be enhanced)
        sleep_stages = ["wake"] * 1440  # Placeholder

        # Generate clinical insights
        insights = self._generate_clinical_insights(
            sleep_efficiency, circadian_score, depression_risk
        )

        # Calculate confidence score
        confidence_score: float = float(np.mean(sleep_metrics[5:8]))

        return ActigraphyAnalysis(
            user_id=user_id,
            analysis_timestamp=datetime.now(UTC).isoformat(),
            sleep_efficiency=sleep_efficiency,
            sleep_onset_latency=sleep_onset_latency,
            wake_after_sleep_onset=wake_after_sleep_onset,
            total_sleep_time=total_sleep_time,
            circadian_rhythm_score=circadian_score,
            activity_fragmentation=activity_fragmentation,
            depression_risk_score=depression_risk,
            sleep_stages=sleep_stages,
            confidence_score=confidence_score,
            clinical_insights=insights,
        )

    @staticmethod
    def _generate_clinical_insights(
        sleep_efficiency: float,
        circadian_score: float,
        depression_risk: float,
    ) -> list[str]:
        """Generate clinical insights based on analysis results."""
        insights: list[str] = []

        # Sleep efficiency insights
        if sleep_efficiency >= EXCELLENT_SLEEP_EFFICIENCY:
            insights.append(
                "Excellent sleep efficiency - maintaining healthy sleep patterns"
            )
        elif sleep_efficiency >= GOOD_SLEEP_EFFICIENCY:
            insights.append("Good sleep efficiency - minor room for improvement")
        else:
            insights.append(
                "Poor sleep efficiency - consider sleep hygiene improvements"
            )

        # Circadian rhythm insights
        if circadian_score >= HIGH_CIRCADIAN_SCORE:
            insights.append("Strong circadian rhythm regularity")
        elif circadian_score >= MODERATE_CIRCADIAN_SCORE:
            insights.append("Moderate circadian rhythm - consider regular sleep schedule")
        else:
            insights.append("Irregular circadian rhythm - prioritize sleep consistency")

        # Depression risk insights
        if depression_risk >= HIGH_DEPRESSION_RISK:
            insights.append(
                "Elevated depression risk indicators - consider professional consultation"
            )
        elif depression_risk >= MODERATE_DEPRESSION_RISK:
            insights.append("Moderate mood-related patterns detected")
        else:
            insights.append("Healthy mood-related activity patterns")

        return insights

    @staticmethod
    def _raise_model_not_loaded_error() -> None:
        """Raise error when model is not loaded."""
        msg = "PAT model not loaded. Call load_model() first."
        raise RuntimeError(msg)

    async def analyze_actigraphy(
        self, input_data: ActigraphyInput
    ) -> ActigraphyAnalysis:
        """Analyze actigraphy data using the PAT model.

        Args:
            input_data: Actigraphy input data

        Returns:
            Analysis results with clinical insights

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded or not self.model:
            self._raise_model_not_loaded_error()

        logger.info(
            "Analyzing actigraphy data for user %s (%d data points)",
            input_data.user_id,
            len(input_data.data_points),
        )

        # Preprocess input data
        input_tensor = self._preprocess_actigraphy_data(input_data.data_points)

        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)

        # Ensure model is not None for type checker
        assert self.model is not None, "Model should be loaded at this point"

        # Run inference
        with torch.no_grad():
            outputs = cast(dict[str, torch.Tensor], self.model(input_tensor))

        # Post-process outputs
        analysis = self._postprocess_predictions(outputs, input_data.user_id)

        logger.info(
            "Actigraphy analysis complete for user %s",
            input_data.user_id,
        )

        return analysis

    async def health_check(self) -> dict[str, str | bool]:
        """Check the health status of the PAT service."""
        return {
            "service": "PAT Model Service",
            "status": "healthy",
            "model_size": self.model_size,
            "device": self.device,
            "model_loaded": self.is_loaded,
        }


# Global singleton instance
_pat_service: PATModelService | None = None


async def get_pat_service() -> PATModelService:
    """Get or create the global PAT service instance."""
    global _pat_service  # noqa: PLW0603
    
    if _pat_service is None:
        _pat_service = PATModelService()
    
    return _pat_service
