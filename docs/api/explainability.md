# AI Explainability API

**UPDATED:** December 6, 2025 - Based on actual implementation and AI integration

## Overview

The Explainability API provides transparency into AI model decisions and insight generation processes within the CLARITY Digital Twin Platform. This is crucial for building trust in AI-generated health recommendations.

## Current Implementation Status

**⚠️ Note**: Full explainability features are planned for future implementation. Current system provides basic transparency through:

- **Insight Source Attribution**: Track which data sources contributed to insights
- **Confidence Scores**: Model confidence levels for all predictions
- **Data Quality Metrics**: Transparency about input data reliability
- **Processing Metadata**: Audit trail of analysis steps

## Available Transparency Features

### Insight Source Attribution

Every AI-generated insight includes source attribution:

```json
{
  "insight_id": "insight-uuid-123",
  "insights": {
    "summary": "Your sleep quality improved significantly...",
    "confidence": 0.87
  },
  "sources": [
    "apple_healthkit_sleep_data",
    "pat_model_sleep_analysis", 
    "gemini_ai_processing",
    "historical_trend_analysis"
  ],
  "data_quality": {
    "completeness": 0.94,
    "reliability": 0.89,
    "confidence": "high"
  },
  "processing_metadata": {
    "model_version": "PAT-M_29k",
    "ai_model": "gemini-2.5-pro",
    "analysis_timestamp": "2025-01-15T10:30:00Z",
    "processing_duration": 23.4
  }
}
```

### Model Confidence Levels

All ML predictions include confidence scores:

**PAT Model Outputs:**
```json
{
  "sleep_analysis": {
    "sleep_stages": [
      {
        "stage": "deep_sleep",
        "start_time": "2025-01-15T00:00:00Z", 
        "duration": 90,
        "confidence": 0.87
      }
    ],
    "confidence": {
      "overall": 0.89,
      "sleep_detection": 0.94,
      "stage_classification": 0.85
    }
  }
}
```

**Gemini AI Insights:**
```json
{
  "recommendations": [
    {
      "category": "sleep",
      "action": "Maintain current bedtime routine",
      "rationale": "Your consistent sleep schedule correlates with improved REM sleep",
      "confidence": 0.83,
      "evidence_strength": "strong"
    }
  ]
}
```

### Data Quality Transparency

Every analysis includes data quality assessment:

```json
{
  "data_quality": {
    "completeness": 0.94,
    "reliability": 0.89, 
    "data_points": 1247,
    "time_coverage": 0.96,
    "sensor_accuracy": 0.92,
    "quality_issues": [
      {
        "type": "missing_data_gap",
        "severity": "minor",
        "impact": "Minimal effect on sleep analysis accuracy"
      }
    ]
  }
}
```

## Planned Explainability Features

### Feature Importance (Future)

*Planned implementation:*

```json
{
  "feature_importance": {
    "sleep_recommendation": {
      "heart_rate_variability": 0.35,
      "movement_patterns": 0.28,
      "historical_sleep_data": 0.22,
      "circadian_rhythm_markers": 0.15
    }
  }
}
```

### Decision Trees (Future)

*Planned visualization of AI reasoning:*

```json
{
  "decision_path": [
    {
      "step": 1,
      "condition": "Average HRV < baseline by 15%",
      "action": "Investigate recovery factors",
      "confidence": 0.91
    },
    {
      "step": 2, 
      "condition": "Sleep efficiency improved by 12%",
      "action": "Maintain current sleep habits",
      "confidence": 0.87
    }
  ]
}
```

### Model Interpretation (Future)

*Planned SHAP (SHapley Additive exPlanations) integration:*

```json
{
  "shap_values": {
    "sleep_quality_prediction": {
      "bedtime_consistency": +0.23,
      "room_temperature": +0.15,
      "caffeine_timing": -0.18,
      "screen_time": -0.12
    }
  }
}
```

## Regulatory Compliance

### Healthcare AI Transparency
- **FDA Guidelines**: Following emerging FDA guidance on AI/ML in medical devices
- **Clinical Decision Support**: Designed for regulatory transparency requirements
- **Audit Trail**: Complete record of AI decision-making process

### Privacy-Preserving Explainability
- **Data Minimization**: Explanations don't expose raw personal data
- **Aggregated Insights**: Focus on patterns, not individual data points
- **User Control**: Users can adjust explanation detail levels

## Implementation Architecture

### Current Components
```
User Request
     ↓
Health Data Analysis (PAT Model)
     ↓
AI Insight Generation (Gemini)
     ↓
Source Attribution & Confidence Scoring
     ↓
Explainable Response
```

### Future Architecture (Planned)
```
User Request
     ↓
Feature Extraction & Importance Scoring
     ↓
Model Prediction with Decision Path Tracking
     ↓
SHAP Value Calculation
     ↓
Natural Language Explanation Generation
     ↓
Interactive Explainability Dashboard
```

## API Usage Examples

### Getting Insight Explanations

```python
import httpx

async with httpx.AsyncClient() as client:
    # Generate insight with explanation
    response = await client.post(
        "http://localhost:8000/api/v1/insights/generate",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "user_id": "user-123",
            "include_explanations": True,
            "explanation_level": "detailed"
        }
    )
    
    insight = response.json()
    
    # Access explanation data
    sources = insight["sources"]
    confidence = insight["data_quality"]["confidence"] 
    processing_info = insight["processing_metadata"]
```

### Confidence Score Interpretation

**Confidence Levels:**
- `0.9-1.0`: Very High - Strong evidence, reliable prediction
- `0.8-0.9`: High - Good evidence, trustworthy prediction
- `0.7-0.8`: Moderate - Fair evidence, reasonable prediction
- `0.6-0.7`: Low - Limited evidence, use with caution
- `0.0-0.6`: Very Low - Insufficient evidence, not recommended

## User Interface Integration

### Dashboard Transparency
- **Confidence Indicators**: Visual confidence meters
- **Source Attribution**: Clear data source labels
- **Quality Scores**: Data reliability indicators
- **Processing Info**: When and how insights were generated

### Interactive Explanations (Future)
- **Drill-down Analysis**: Explore specific recommendation reasoning
- **What-if Scenarios**: See how different inputs affect recommendations
- **Historical Explanation**: Track how explanations change over time

## Testing & Validation

### Explanation Quality Tests
- **Consistency**: Same inputs produce consistent explanations
- **Accuracy**: Explanations accurately reflect model behavior
- **Completeness**: All major factors included in explanations
- **Understandability**: Non-technical users can comprehend explanations

### Implementation Details
- **Location**: Distributed across API endpoints (transparency features)
- **Future Location**: `src/clarity/api/v1/explainability.py` (planned)
- **Dependencies**: SHAP, LIME (planned), custom attribution logic
- **Testing**: Explanation validation tests in `tests/explainability/` (planned)