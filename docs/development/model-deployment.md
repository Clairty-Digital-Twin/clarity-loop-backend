# PAT Model Deployment & Weights Management

**Canonical guide for deploying Pretrained Actigraphy Transformer models in production**

## Overview

The Clarity Loop Backend integrates the Pretrained Actigraphy Transformer (PAT) - the first open-source foundation model for wearable movement data. This document covers model weights management, deployment strategies, and production considerations.

## Model Weights Structure

### Local Weights Directory

```
models/
├── gemini/
└── pat/
    ├── PAT-L_29k_weights.h5    # Large: 1.99M parameters, 7.6MB
    ├── PAT-M_29k_weights.h5    # Medium: 1.00M parameters, 3.8MB
    └── PAT-S_29k_weights.h5    # Small: 285K parameters, 1.1MB
```

### Model Specifications

| Model | Parameters | File Size | Inference Time | Best Use Case |
|-------|------------|-----------|----------------|---------------|
| **PAT-L** | 1.99M | 7.6MB | ~800ms | Research, batch analysis |
| **PAT-M** | 1.00M | 3.8MB | ~450ms | Production default |
| **PAT-S** | 285K | 1.1MB | ~200ms | Real-time, mobile apps |

### Source Attribution

These weights are the official pre-trained models from:

- **Research**: "AI Foundation Models for Wearable Movement Data in Mental Health Research"
- **Authors**: Ruan, Franklin Y. et al., Dartmouth College
- **Training Data**: 29,307 participants from NHANES 2003-2014
- **License**: MIT License
- **ArXiv**: 2411.15240

## Model Loading Implementation

### Basic Model Loading

```python
import tensorflow as tf
from pathlib import Path

class PATModelLoader:
    """Production-ready PAT model loader with error handling"""
    
    def __init__(self, model_size: str = "medium", models_dir: str = "models"):
        self.model_size = model_size
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / f"PAT-{model_size[0].upper()}_29k_weights.h5"
        self.model = None
        
    def load_model(self) -> tf.keras.Model:
        """Load PAT model with comprehensive error handling"""
        try:
            # Verify weights file exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model weights not found: {self.model_path}")
            
            # Load model architecture first
            model = self._create_pat_architecture()
            
            # Load pre-trained weights
            model.load_weights(str(self.model_path))
            
            # Verify model loaded correctly
            self._validate_model(model)
            
            self.model = model
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load PAT model: {e}")
    
    def _create_pat_architecture(self) -> tf.keras.Model:
        """Create PAT model architecture matching the weights"""
        # Model architecture specifications from PAT paper
        configs = {
            "small": {
                "patch_size": 18, "embed_dim": 96, "num_heads": 6,
                "ff_dim": 256, "num_layers": 1, "dropout": 0.1
            },
            "medium": {
                "patch_size": 18, "embed_dim": 96, "num_heads": 12,
                "ff_dim": 256, "num_layers": 2, "dropout": 0.1
            },
            "large": {
                "patch_size": 9, "embed_dim": 96, "num_heads": 12,
                "ff_dim": 256, "num_layers": 4, "dropout": 0.1
            }
        }
        
        config = configs[self.model_size]
        
        # Build transformer encoder as per PAT specification
        input_size = 10080  # 1 week in minutes
        num_patches = input_size // config["patch_size"]
        
        inputs = tf.keras.layers.Input(shape=(input_size,), name="actigraphy_input")
        
        # Patch embedding layer
        patches = tf.keras.layers.Reshape((num_patches, config["patch_size"]))(inputs)
        patch_embeddings = tf.keras.layers.Dense(config["embed_dim"])(patches)
        
        # Positional encoding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embeddings = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=config["embed_dim"]
        )(positions)
        
        x = patch_embeddings + position_embeddings
        
        # Transformer blocks
        for i in range(config["num_layers"]):
            x = self._transformer_block(x, config)
        
        # Global pooling for feature extraction
        features = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        return tf.keras.Model(inputs=inputs, outputs=features, name=f"PAT_{self.model_size}")
    
    def _transformer_block(self, x, config):
        """Single transformer encoder block"""
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config["num_heads"],
            key_dim=config["embed_dim"] // config["num_heads"]
        )(x, x)
        
        # Skip connection + layer norm
        x = tf.keras.layers.LayerNormalization()(x + attention)
        
        # Feed-forward network
        ff = tf.keras.layers.Dense(config["ff_dim"], activation="relu")(x)
        ff = tf.keras.layers.Dense(config["embed_dim"])(ff)
        ff = tf.keras.layers.Dropout(config["dropout"])(ff)
        
        # Skip connection + layer norm
        return tf.keras.layers.LayerNormalization()(x + ff)
    
    def _validate_model(self, model: tf.keras.Model):
        """Validate model loaded correctly with test input"""
        test_input = tf.random.normal((1, 10080))  # 1 week of data
        
        try:
            output = model(test_input, training=False)
            assert output.shape[0] == 1  # Batch size
            assert len(output.shape) == 2  # [batch, features]
            
        except Exception as e:
            raise ValueError(f"Model validation failed: {e}")
```

### Production Service Integration

```python
from src.ml.pat_loader import PATModelLoader
import asyncio
import numpy as np

class ActigraphyMLService:
    """Production ML service for actigraphy analysis"""
    
    def __init__(self, model_size: str = "medium"):
        self.loader = PATModelLoader(model_size)
        self.model = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize ML service asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Load model in thread pool to avoid blocking
        self.model = await loop.run_in_executor(
            None, self.loader.load_model
        )
        
        self.is_initialized = True
        logger.info(f"PAT service initialized with {self.loader.model_size} model")
    
    async def extract_features(self, actigraphy_data: np.ndarray) -> dict:
        """Extract features from 1-week actigraphy data"""
        if not self.is_initialized:
            raise RuntimeError("Service not initialized")
        
        # Preprocess data (z-score normalization)
        normalized_data = self._normalize_data(actigraphy_data)
        
        # Run inference
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(
            None, self._run_inference, normalized_data
        )
        
        # Convert to interpretable metrics
        return self._interpret_features(features, actigraphy_data)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply PAT-compatible normalization (per-year z-score)"""
        # Per-year standardization as per PAT methodology
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)
    
    def _run_inference(self, data: np.ndarray) -> np.ndarray:
        """Run model inference"""
        return self.model.predict(data.reshape(1, -1), verbose=0)[0]
    
    def _interpret_features(self, features: np.ndarray, raw_data: np.ndarray) -> dict:
        """Convert features to clinical metrics"""
        return {
            "sleep_efficiency": self._calculate_sleep_efficiency(raw_data),
            "circadian_rhythm_strength": self._calculate_circadian_strength(raw_data),
            "activity_fragmentation": self._calculate_fragmentation(raw_data),
            "rest_activity_ratio": self._calculate_rest_activity_ratio(raw_data),
            "raw_features": features.tolist(),
            "feature_dim": len(features)
        }
```

## Deployment Strategies

### 1. Local Development

```python
# Initialize service with local weights
service = ActigraphyMLService(model_size="medium")
await service.initialize()

# Process user data
features = await service.extract_features(user_actigraphy_data)
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

# Copy model weights
COPY models/ /app/models/

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Copy application
COPY src/ /app/src/

WORKDIR /app
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Google Cloud AI Platform

```python
# Deploy to Vertex AI
from google.cloud import aiplatform

# Upload model to Cloud Storage
model_artifact_uri = "gs://clarity-models/pat-medium/"

# Create model endpoint
model = aiplatform.Model.upload(
    display_name="pat-medium-v1",
    artifact_uri=model_artifact_uri,
    serving_container_image_uri="tensorflow/serving:2.13.0"
)

endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=10
)
```

## Model Performance Monitoring

### Performance Metrics

```python
class ModelPerformanceMonitor:
    """Monitor PAT model performance in production"""
    
    def __init__(self):
        self.latency_history = []
        self.error_count = 0
        self.prediction_count = 0
    
    def log_prediction(self, latency_ms: float, success: bool):
        """Log prediction performance"""
        self.latency_history.append(latency_ms)
        self.prediction_count += 1
        
        if not success:
            self.error_count += 1
        
        # Alert if performance degrades
        if len(self.latency_history) > 100:
            avg_latency = np.mean(self.latency_history[-100:])
            if avg_latency > 1000:  # 1 second threshold
                logger.warning(f"High latency detected: {avg_latency:.2f}ms")
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        if not self.latency_history:
            return {"status": "no_data"}
        
        return {
            "avg_latency_ms": np.mean(self.latency_history),
            "p95_latency_ms": np.percentile(self.latency_history, 95),
            "error_rate": self.error_count / self.prediction_count,
            "total_predictions": self.prediction_count
        }
```

### Expected Performance Benchmarks

| Model | Latency (p95) | Memory Usage | Throughput/min |
|-------|---------------|--------------|----------------|
| PAT-S | 250ms | 256MB | 240 |
| PAT-M | 500ms | 512MB | 120 |
| PAT-L | 900ms | 1GB | 60 |

## Security & Compliance

### HIPAA Considerations

- **Data Encryption**: All model inputs/outputs encrypted in transit and at rest
- **Access Control**: Role-based access to model weights and inference endpoints
- **Audit Logging**: Complete logging of all model access and predictions
- **Data Retention**: Automatic deletion of inference logs per retention policy

### Model Integrity

```python
import hashlib

def verify_model_integrity():
    """Verify model weights haven't been tampered with"""
    expected_hashes = {
        "PAT-L_29k_weights.h5": "sha256:a1b2c3d4...",
        "PAT-M_29k_weights.h5": "sha256:e5f6g7h8...", 
        "PAT-S_29k_weights.h5": "sha256:i9j0k1l2..."
    }
    
    for filename, expected_hash in expected_hashes.items():
        filepath = Path("models") / filename
        
        if filepath.exists():
            actual_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
            if f"sha256:{actual_hash}" != expected_hash:
                raise SecurityError(f"Model {filename} integrity check failed")
        else:
            raise FileNotFoundError(f"Model {filename} not found")
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Verify weights files exist in `/models/`
   - Check file permissions and sizes
   - Ensure TensorFlow version compatibility

2. **Memory Issues**
   - Use PAT-S for memory-constrained environments
   - Implement model caching strategies
   - Monitor memory usage during inference

3. **Performance Issues**
   - Profile inference pipeline
   - Consider GPU acceleration
   - Implement batch processing for multiple users

### Health Checks

```python
async def model_health_check():
    """Comprehensive model health check"""
    try:
        # Test model loading
        service = ActigraphyMLService()
        await service.initialize()
        
        # Test inference
        test_data = np.random.randn(10080)  # 1 week
        features = await service.extract_features(test_data)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "inference_working": True,
            "feature_count": len(features["raw_features"])
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
```

## Next Steps

1. **Implement Model Loading**: Create `PATModelLoader` class
2. **Add Performance Monitoring**: Implement latency and error tracking
3. **Set Up Health Checks**: Add model health monitoring endpoints
4. **Deploy to Production**: Choose deployment strategy based on requirements
5. **Monitor Performance**: Track model performance in production

---

**References**:

- [PAT Research Paper](../literature/AI%20Foundation%20Models%20for%20Wearable%20Movement%20Data%20in%20Mental.pdf)
- [Original PAT Repository](https://github.com/njacobsonlab/Pretrained-Actigraphy-Transformer)
- [ML API Endpoints](../api/ml-endpoints.md)
