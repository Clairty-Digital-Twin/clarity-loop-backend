# Explainability API

This document provides comprehensive documentation for the AI explainability endpoints in the Clarity Loop Backend API, enabling transparent and interpretable machine learning predictions.

## Overview

The Explainability API provides attention visualizations and feature importance analysis from the Actigraphy Transformer model. This enables users and healthcare providers to understand which temporal patterns and data features contributed to specific health predictions and insights.

## Core Concepts

### Attention Mechanisms
The PAT (Pretrained Actigraphy Transformer) model uses multi-head attention to focus on relevant temporal patterns in actigraphy data. The explainability API exposes these attention weights to show:

- **Temporal Attention**: Which time periods were most important for predictions
- **Feature Importance**: Which sensor measurements drove specific classifications
- **Pattern Recognition**: How weekly and daily rhythms influenced outcomes

### Visualization Types
- **Attention Heatmaps**: 2D visualizations of attention weights across time
- **Feature Attribution**: Quantified importance scores for different data modalities
- **Temporal Patterns**: Time-series plots showing critical decision periods
- **Weekly Rhythms**: Circadian and circaseptian pattern analysis

## Explainability Endpoints

### Get Attention Analysis

Retrieve attention matrices and feature importance for a specific health prediction.

#### Request
```http
GET /v1/actigraphy/{jobId}/explain
Authorization: Bearer <firebase-jwt-token>
```

#### Query Parameters
- `visualization_type` (optional): `heatmap`, `attribution`, `temporal`, `weekly`
- `format` (optional): `json`, `png`, `svg`
- `resolution` (optional): `high`, `medium`, `low` - affects PNG/SVG detail level
- `time_range` (optional): `1d`, `7d`, `30d` - temporal scope for analysis

#### Response
```json
{
  "success": true,
  "data": {
    "job_id": "actig_job_20240120_abc123",
    "model_version": "pat-l-v1.2",
    "explanation_timestamp": "2024-01-20T10:30:00Z",
    "attention_analysis": {
      "encoder_attention": {
        "shape": [12, 560, 560],
        "description": "Multi-head attention weights from encoder layers",
        "aggregation_method": "mean_across_heads",
        "normalized_weights": "softmax_per_head"
      },
      "decoder_attention": {
        "shape": [8, 560, 128],
        "description": "Cross-attention weights from decoder to encoder",
        "aggregation_method": "max_across_heads"
      },
      "temporal_importance": {
        "time_patches": [
          {
            "patch_id": 0,
            "time_range": "2024-01-13T00:00:00Z to 2024-01-13T00:18:00Z",
            "importance_score": 0.85,
            "attention_weight": 0.023,
            "contributing_features": ["activity_count", "heart_rate_variance"]
          },
          {
            "patch_id": 1,
            "time_range": "2024-01-13T00:18:00Z to 2024-01-13T00:36:00Z", 
            "importance_score": 0.12,
            "attention_weight": 0.003,
            "contributing_features": ["sleep_stage"]
          }
        ]
      }
    },
    "feature_attribution": {
      "activity_count": {
        "global_importance": 0.34,
        "confidence": 0.92,
        "interpretation": "High activity variance during evening hours significantly influenced depression risk assessment"
      },
      "heart_rate": {
        "global_importance": 0.28,
        "confidence": 0.87,
        "interpretation": "Elevated resting heart rate patterns during sleep indicate stress response"
      },
      "sleep_efficiency": {
        "global_importance": 0.23,
        "confidence": 0.91,
        "interpretation": "Fragmented sleep patterns correlate with mood regulation difficulties"
      }
    },
    "visualizations": {
      "attention_heatmap_url": "https://storage.googleapis.com/clarity-explanations/attention_heatmap_20240120.png",
      "temporal_plot_url": "https://storage.googleapis.com/clarity-explanations/temporal_importance_20240120.png",
      "feature_attribution_url": "https://storage.googleapis.com/clarity-explanations/feature_attribution_20240120.png"
    }
  }
}
```

### Generate Explanation Report

Create a comprehensive explanation document for a health prediction.

#### Request
```http
POST /v1/actigraphy/{jobId}/explain/report
Content-Type: application/json
Authorization: Bearer <firebase-jwt-token>
```

```json
{
  "report_type": "clinical_summary",
  "include_visualizations": true,
  "target_audience": "healthcare_provider",
  "detail_level": "comprehensive",
  "time_granularity": "hourly"
}
```

#### Response
```json
{
  "success": true,
  "data": {
    "report_id": "exp_report_20240120_xyz789",
    "generation_timestamp": "2024-01-20T10:45:00Z",
    "report_url": "https://storage.googleapis.com/clarity-reports/clinical_summary_20240120.pdf",
    "key_findings": [
      {
        "finding": "Sleep pattern disruption",
        "evidence": "Attention weights concentrated on 2-4 AM periods with elevated heart rate",
        "clinical_relevance": "Indicates potential sleep disorder requiring further evaluation",
        "confidence": 0.89
      },
      {
        "finding": "Circadian rhythm irregularity", 
        "evidence": "Weekly attention pattern shows inconsistent sleep-wake cycles",
        "clinical_relevance": "May contribute to mood regulation difficulties",
        "confidence": 0.76
      }
    ],
    "recommendations": [
      "Sleep hygiene assessment recommended",
      "Consider circadian rhythm therapy",
      "Monitor stress levels during identified high-attention periods"
    ]
  }
}
```

## Attention Matrix Processing (PAT Implementation)

### Normalization and Aggregation Rules

Based on PAT research implementation (pp 31-33):

```python
def process_attention_matrices(attention_weights, aggregation_method="mean_across_heads"):
    """
    Process raw attention matrices from PAT model for explainability
    
    Args:
        attention_weights: Raw attention tensor [batch, heads, seq_len, seq_len]
        aggregation_method: How to combine multi-head attention
    
    Returns:
        Processed attention matrices with normalization
    """
    
    if aggregation_method == "mean_across_heads":
        # Average attention across all heads (standard approach)
        aggregated = attention_weights.mean(dim=1)
    elif aggregation_method == "max_across_heads":
        # Take maximum attention across heads (highlights dominant patterns)
        aggregated = attention_weights.max(dim=1)[0]
    elif aggregation_method == "entropy_weighted":
        # Weight by attention entropy (focus on confident decisions)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-9), dim=-1)
        weights = torch.softmax(-entropy, dim=1)  # Lower entropy = higher weight
        aggregated = torch.sum(attention_weights * weights.unsqueeze(-1), dim=1)
    
    # Apply row-wise normalization to ensure attention sums to 1
    aggregated = torch.softmax(aggregated, dim=-1)
    
    # Apply temporal smoothing to reduce noise (optional)
    if apply_temporal_smoothing:
        kernel = torch.ones(1, 1, 3) / 3.0  # Simple 3-point average
        aggregated = F.conv1d(aggregated.unsqueeze(1), kernel, padding=1).squeeze(1)
    
    return aggregated

def map_attention_to_time_periods(attention_matrix, patch_timestamps):
    """
    Map attention weights back to real-world time periods
    
    Args:
        attention_matrix: Processed attention weights [seq_len, seq_len]
        patch_timestamps: List of timestamp tuples for each patch
    
    Returns:
        Time-indexed attention analysis
    """
    temporal_importance = []
    
    for i, (start_time, end_time) in enumerate(patch_timestamps):
        # Sum attention weights for this time patch
        patch_attention = attention_matrix[:, i].sum().item()
        
        # Calculate importance score (normalized across all patches)
        importance_score = patch_attention / attention_matrix.sum().item()
        
        temporal_importance.append({
            "patch_id": i,
            "time_range": f"{start_time} to {end_time}",
            "importance_score": float(importance_score),
            "attention_weight": float(patch_attention),
            "contributing_features": identify_contributing_features(attention_matrix, i)
        })
    
    return sorted(temporal_importance, key=lambda x: x["importance_score"], reverse=True)
```

### Visualization Generation

```python
def generate_attention_heatmap(attention_matrix, time_labels, output_format="png"):
    """
    Generate attention heatmap visualization
    
    Args:
        attention_matrix: Processed attention weights [seq_len, seq_len]
        time_labels: Human-readable time labels for patches
        output_format: Output format (png, svg, or json)
    
    Returns:
        Visualization data or file path
    """
    
    if output_format == "json":
        return {
            "matrix": attention_matrix.tolist(),
            "time_labels": time_labels,
            "colormap": "viridis",
            "title": "Temporal Attention Patterns"
        }
    
    elif output_format in ["png", "svg"]:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            attention_matrix.cpu().numpy(),
            xticklabels=time_labels,
            yticklabels=time_labels,
            cmap="viridis",
            annot=False,
            cbar_kws={"label": "Attention Weight"}
        )
        plt.title("PAT Model Temporal Attention Analysis")
        plt.xlabel("Target Time Period")
        plt.ylabel("Source Time Period")
        
        # Save to Google Cloud Storage
        filename = f"attention_heatmap_{uuid.uuid4().hex}.{output_format}"
        storage_path = upload_to_gcs(plt, filename, format=output_format)
        plt.close()
        
        return storage_path
```

## Security and Privacy

### Data Protection
- **Attention matrices**: Aggregated and anonymized before storage
- **Temporal patterns**: Relative patterns only, no absolute timestamps in logs
- **Feature attribution**: Statistical summaries only, no raw health data exposed
- **Visualization storage**: Temporary URLs with 24-hour expiration

### Access Control
- **Healthcare providers**: Full access to clinical-grade explanations
- **Patients**: Simplified explanations with encouraging framing
- **Researchers**: Anonymized aggregate attention patterns only
- **Audit logging**: All explanation requests logged for compliance

## Performance Specifications

### Response Times
- **Attention matrix retrieval**: < 500ms (cached)
- **Real-time explanation generation**: < 2 seconds
- **Comprehensive report generation**: < 30 seconds
- **Visualization rendering**: < 5 seconds

### Caching Strategy
- **Attention matrices**: 24-hour cache for completed jobs
- **Visualizations**: 7-day cache in Google Cloud Storage
- **Reports**: 30-day cache for clinical documentation
- **Feature attributions**: 1-hour cache for real-time updates

## Error Handling

### Common Error Responses

#### Model Explanation Not Available
```json
{
  "success": false,
  "error": {
    "code": "EXPLANATION_NOT_AVAILABLE",
    "message": "Attention analysis not available for this prediction",
    "details": {
      "reason": "Model version does not support explainability",
      "suggested_action": "Re-run analysis with explainable model version"
    }
  }
}
```

#### Visualization Generation Failed
```json
{
  "success": false,
  "error": {
    "code": "VISUALIZATION_ERROR",
    "message": "Failed to generate attention visualization",
    "details": {
      "reason": "Insufficient data for meaningful visualization",
      "min_required_patches": 168,
      "available_patches": 45
    }
  }
}
```

## Rate Limiting

### Explanation Request Limits
- **Per user**: 50 explanation requests per hour
- **Per job**: 5 different visualization types per job
- **Report generation**: 10 comprehensive reports per day
- **Burst allowance**: 10 requests per minute

## Integration Examples

### Frontend Integration
```javascript
// Request attention analysis for a completed job
async function getAttentionAnalysis(jobId, visualizationType = 'heatmap') {
  const response = await fetch(`/v1/actigraphy/${jobId}/explain?visualization_type=${visualizationType}`, {
    headers: {
      'Authorization': `Bearer ${firebaseToken}`,
      'Content-Type': 'application/json'
    }
  });
  
  const data = await response.json();
  
  if (data.success) {
    // Display attention heatmap in UI
    displayAttentionVisualization(data.data.visualizations.attention_heatmap_url);
    
    // Show feature importance
    renderFeatureAttribution(data.data.feature_attribution);
  }
}

// Generate clinical report
async function generateClinicalReport(jobId) {
  const response = await fetch(`/v1/actigraphy/${jobId}/explain/report`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${firebaseToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      report_type: 'clinical_summary',
      include_visualizations: true,
      target_audience: 'healthcare_provider',
      detail_level: 'comprehensive'
    })
  });
  
  const data = await response.json();
  if (data.success) {
    window.open(data.data.report_url, '_blank');
  }
}
```

### Clinical Dashboard Integration
```python
# Python integration for healthcare provider dashboards
import requests
from typing import Dict, List

class ClarityExplainabilityClient:
    def __init__(self, api_base_url: str, auth_token: str):
        self.base_url = api_base_url
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
    
    def get_patient_explanation(self, job_id: str) -> Dict:
        """Get attention analysis for clinical review"""
        response = requests.get(
            f"{self.base_url}/v1/actigraphy/{job_id}/explain",
            headers=self.headers,
            params={'visualization_type': 'clinical', 'format': 'json'}
        )
        return response.json()
    
    def generate_clinical_summary(self, job_id: str, patient_context: Dict) -> str:
        """Generate comprehensive clinical explanation report"""
        payload = {
            'report_type': 'clinical_summary',
            'include_visualizations': True,
            'target_audience': 'healthcare_provider',
            'detail_level': 'comprehensive',
            'patient_context': patient_context
        }
        
        response = requests.post(
            f"{self.base_url}/v1/actigraphy/{job_id}/explain/report",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()['data']['report_url']
        else:
            raise Exception(f"Report generation failed: {response.text}")
```
