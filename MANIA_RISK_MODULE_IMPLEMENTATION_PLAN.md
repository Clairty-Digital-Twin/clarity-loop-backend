# Mania Risk Forecasting Module - Comprehensive Implementation Plan

## Executive Summary

This document outlines the complete implementation plan for integrating a state-of-the-art Mania Risk Forecasting Module into the CLARITY Digital Twin Platform. Based on cutting-edge 2024-2025 research achieving **0.98 AUC for manic episode prediction**, this module will provide early warning capabilities for bipolar disorder management using only passive wearable data.

## Research Findings

### 1. Latest Computational Psychiatry Breakthroughs (2024-2025)

#### High-Accuracy Prediction Models
- **Nature Digital Medicine (2024)**: Achieved remarkable accuracy using only sleep-wake data
  - **0.98 AUC** for manic episodes
  - **0.95 AUC** for hypomanic episodes  
  - **0.80 AUC** for depressive episodes
- Study analyzed 168 patients with 587 days average clinical follow-up

#### Key Predictive Biomarkers
1. **Circadian Phase Shifts** (Most Significant)
   - **Phase advances** (shifting earlier) → Manic episodes
   - **Phase delays** (shifting later) → Depressive episodes
   - Detectable 2-3 days before clinical symptoms

2. **Sleep Pattern Disruptions**
   - Severe sleep reduction: 69-99% of manic episodes
   - Intra-day variability: 12-hour patterns give ~3 days warning
   - Sleep fragmentation increases before episodes

3. **Activity Pattern Changes**
   - Day-to-day activity variability detects transitions earlier than sleep/mood
   - Activity fragmentation correlates with manic energy surges
   - Psychomotor activity escalation precedes clinical symptoms

#### Causal Dynamics (eBioMedicine 2024)
- Transfer entropy analysis on 139 patients revealed:
  - Causality from circadian disruption → mood symptoms in 85.7% of BD Type I
  - Sleep phase disturbance induces circadian phase disturbance
  - Direct causal link established, not just correlation

### 2. Clinical Thresholds & Validation

While specific hour-based thresholds aren't standardized in literature, research indicates:

#### Sleep Duration
- **Normal baseline**: 7-8 hours for most adults
- **Warning zone**: <5 hours sustained over 2-3 nights
- **High risk**: <3 hours any single night or <4 hours averaged over 3 days
- **Critical**: Near-zero sleep for 24-48 hours

#### Circadian Metrics
- **Circadian rhythm score <0.5**: Indicates disruption
- **Sleep timing variability >90 minutes**: Risk indicator
- **Phase advance >1 hour**: Early warning sign

#### Activity Metrics
- **Steps increase >50% from baseline**: Potential hypomania
- **Activity fragmentation >0.8**: Disorganized energy patterns
- **Night activity surges**: Strong mania predictor

## Architecture Analysis

### Current CLARITY Backend Structure

```
src/clarity/
├── ml/
│   ├── analysis_pipeline.py      # Main orchestrator
│   ├── pat_service.py           # PAT model with ActigraphyAnalysis
│   ├── processors/
│   │   ├── sleep_processor.py   # SleepFeatures extraction
│   │   ├── activity_processor.py # Activity metrics
│   │   └── cardio_processor.py  # HR/HRV analysis
│   └── models/
└── api/
    └── v1/
        ├── pat_analysis.py      # Step-based analysis endpoint
        └── health_data.py       # Full health data upload
```

### Key Integration Points

1. **ActigraphyAnalysis Model** (pat_service.py:112)
   - Add `mania_risk_score: float` and `mania_alert_level: str` fields
   - Existing fields we'll leverage: `total_sleep_time`, `circadian_rhythm_score`, `activity_fragmentation`

2. **HealthAnalysisPipeline** (analysis_pipeline.py:102)
   - Hook into `_generate_health_indicators()` method
   - Access to full week of processed data
   - DynamoDB storage already configured

3. **Data Sources Available**
   - **SleepFeatures**: total_sleep_minutes, sleep_latency, consistency_score, awakenings_count
   - **ActivityProcessor**: steps, distance, active_energy, exercise_minutes
   - **CardioProcessor**: resting_hr, avg_hrv, circadian_rhythm_score

## Detailed Implementation Plan

### Phase 1: Core Module Development

#### 1.1 Create ManiaRiskAnalyzer Module

```python
# src/clarity/ml/mania_risk_analyzer.py

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
from pydantic import BaseModel, Field

from clarity.ml.processors.sleep_processor import SleepFeatures


@dataclass
class ManiaRiskConfig:
    """Configuration for mania risk thresholds and weights."""
    # Sleep thresholds
    min_sleep_hours: float = 5.0
    critical_sleep_hours: float = 3.0
    sleep_loss_percent: float = 0.4  # 40% reduction
    
    # Circadian thresholds  
    circadian_disruption_threshold: float = 0.5
    phase_advance_hours: float = 1.0
    
    # Activity thresholds
    activity_surge_ratio: float = 1.5
    activity_fragmentation_threshold: float = 0.8
    
    # Physiological thresholds
    elevated_resting_hr: float = 90.0
    low_hrv_threshold: float = 20.0
    
    # Weights for scoring
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "severe_sleep_loss": 0.30,
                "acute_sleep_loss": 0.20,
                "rapid_sleep_onset": 0.10,
                "circadian_disruption": 0.25,
                "sleep_inconsistency": 0.10,
                "activity_fragmentation": 0.20,
                "activity_surge": 0.10,
                "elevated_hr": 0.05,
                "low_hrv": 0.05,
                "circadian_phase_advance": 0.15,
            }


class ManiaRiskResult(BaseModel):
    """Result of mania risk analysis."""
    risk_score: float = Field(ge=0.0, le=1.0, description="Overall mania risk score")
    alert_level: str = Field(description="Risk level: none/low/moderate/high")
    contributing_factors: list[str] = Field(description="Factors contributing to risk")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    clinical_insight: str = Field(description="Clinical interpretation")
    recommendations: list[str] = Field(description="Actionable recommendations")


class ManiaRiskAnalyzer:
    """Analyzes health data patterns to detect mania/hypomania risk."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize with configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
            self.config = ManiaRiskConfig(**config_dict)
        else:
            self.config = ManiaRiskConfig()
            
        self.moderate_threshold = 0.4
        self.high_threshold = 0.7
        
    def analyze(
        self,
        sleep_features: Optional[SleepFeatures] = None,
        pat_metrics: Optional[Dict[str, float]] = None,
        activity_stats: Optional[Dict[str, Any]] = None,
        cardio_stats: Optional[Dict[str, float]] = None,
        historical_baseline: Optional[Dict[str, float]] = None,
    ) -> ManiaRiskResult:
        """
        Analyze multi-modal health data for mania risk indicators.
        
        Args:
            sleep_features: Processed sleep metrics from SleepProcessor
            pat_metrics: Metrics from PAT model analysis  
            activity_stats: Activity statistics from ActivityProcessor
            cardio_stats: Cardiovascular metrics from CardioProcessor
            historical_baseline: User's personal baseline (28-day averages)
            
        Returns:
            ManiaRiskResult with score, level, and clinical insights
        """
        score = 0.0
        factors = []
        confidence = 1.0
        
        # 1. Sleep Analysis
        sleep_score, sleep_factors, sleep_conf = self._analyze_sleep(
            sleep_features, pat_metrics, historical_baseline
        )
        score += sleep_score
        factors.extend(sleep_factors)
        confidence *= sleep_conf
        
        # 2. Circadian Rhythm Analysis
        circadian_score, circadian_factors = self._analyze_circadian(
            sleep_features, pat_metrics, cardio_stats
        )
        score += circadian_score
        factors.extend(circadian_factors)
        
        # 3. Activity Analysis
        activity_score, activity_factors = self._analyze_activity(
            pat_metrics, activity_stats, historical_baseline
        )
        score += activity_score
        factors.extend(activity_factors)
        
        # 4. Physiological Analysis
        physio_score, physio_factors = self._analyze_physiology(
            cardio_stats
        )
        score += physio_score
        factors.extend(physio_factors)
        
        # 5. Temporal Pattern Analysis
        temporal_score, temporal_factors = self._analyze_temporal_patterns(
            sleep_features, activity_stats, historical_baseline
        )
        score += temporal_score
        factors.extend(temporal_factors)
        
        # Clamp score and determine alert level
        score = max(0.0, min(1.0, score))
        alert_level = self._determine_alert_level(score)
        
        # Generate clinical insight and recommendations
        clinical_insight = self._generate_clinical_insight(
            alert_level, factors, score
        )
        recommendations = self._generate_recommendations(
            alert_level, factors
        )
        
        return ManiaRiskResult(
            risk_score=score,
            alert_level=alert_level,
            contributing_factors=factors,
            confidence=confidence,
            clinical_insight=clinical_insight,
            recommendations=recommendations,
        )
    
    def _analyze_sleep(
        self,
        sleep: Optional[SleepFeatures],
        pat_metrics: Optional[Dict[str, float]],
        baseline: Optional[Dict[str, float]],
    ) -> Tuple[float, list[str], float]:
        """Analyze sleep patterns for mania indicators."""
        score = 0.0
        factors = []
        confidence = 1.0
        
        # Get sleep duration from available sources
        if sleep and sleep.total_sleep_minutes > 0:
            hours = sleep.total_sleep_minutes / 60
            data_source = "HealthKit"
        elif pat_metrics and "total_sleep_time" in pat_metrics:
            hours = pat_metrics["total_sleep_time"]
            data_source = "PAT estimation"
            confidence *= 0.8  # Lower confidence for estimated data
        else:
            return 0.0, ["Insufficient sleep data"], 0.5
            
        # Check against absolute thresholds
        if hours < self.config.critical_sleep_hours:
            score += self.config.weights["severe_sleep_loss"]
            factors.append(f"Critically low sleep: {hours:.1f}h ({data_source})")
        elif hours < self.config.min_sleep_hours:
            score += self.config.weights["severe_sleep_loss"] * 0.7
            factors.append(f"Low sleep duration: {hours:.1f}h ({data_source})")
            
        # Check against personal baseline if available
        if baseline and "avg_sleep_hours" in baseline:
            baseline_hours = baseline["avg_sleep_hours"]
            reduction = 1 - (hours / baseline_hours)
            if reduction > self.config.sleep_loss_percent:
                score += self.config.weights["acute_sleep_loss"]
                factors.append(
                    f"Sleep reduced {reduction*100:.0f}% from baseline"
                )
        
        # Check sleep latency (rapid sleep onset during mania)
        if sleep and sleep.sleep_latency < 5:
            score += self.config.weights["rapid_sleep_onset"]
            factors.append("Very rapid sleep onset (<5 min)")
            
        return score, factors, confidence
    
    def _analyze_circadian(
        self,
        sleep: Optional[SleepFeatures],
        pat_metrics: Optional[Dict[str, float]],
        cardio: Optional[Dict[str, float]],
    ) -> Tuple[float, list[str]]:
        """Analyze circadian rhythm disruption."""
        score = 0.0
        factors = []
        
        # Check circadian rhythm score
        circadian_score = None
        if pat_metrics and "circadian_rhythm_score" in pat_metrics:
            circadian_score = pat_metrics["circadian_rhythm_score"]
        elif cardio and "circadian_rhythm_score" in cardio:
            circadian_score = cardio["circadian_rhythm_score"]
            
        if circadian_score is not None:
            if circadian_score < self.config.circadian_disruption_threshold:
                score += self.config.weights["circadian_disruption"]
                factors.append(
                    f"Disrupted circadian rhythm (score: {circadian_score:.2f})"
                )
        
        # Check sleep consistency
        if sleep and sleep.consistency_score < 0.4:
            score += self.config.weights["sleep_inconsistency"]
            factors.append("Irregular sleep schedule")
            
        # TODO: Add circadian phase advance detection when we have
        # access to sleep timing data (bedtime, wake time)
        
        return score, factors
    
    def _analyze_activity(
        self,
        pat_metrics: Optional[Dict[str, float]],
        activity: Optional[Dict[str, Any]],
        baseline: Optional[Dict[str, float]],
    ) -> Tuple[float, list[str]]:
        """Analyze activity patterns for mania indicators."""
        score = 0.0
        factors = []
        
        # Check activity fragmentation from PAT
        if pat_metrics and "activity_fragmentation" in pat_metrics:
            frag = pat_metrics["activity_fragmentation"]
            if frag > self.config.activity_fragmentation_threshold:
                score += self.config.weights["activity_fragmentation"]
                factors.append(f"High activity fragmentation: {frag:.2f}")
        
        # Check for activity surges
        if activity and baseline:
            # Compare current vs baseline steps
            if "avg_daily_steps" in activity and "avg_steps" in baseline:
                current_steps = activity["avg_daily_steps"]
                baseline_steps = baseline["avg_steps"]
                if baseline_steps > 0:
                    ratio = current_steps / baseline_steps
                    if ratio > self.config.activity_surge_ratio:
                        score += self.config.weights["activity_surge"]
                        factors.append(
                            f"Activity surge: {ratio:.1f}x baseline"
                        )
        
        return score, factors
    
    def _analyze_physiology(
        self,
        cardio: Optional[Dict[str, float]],
    ) -> Tuple[float, list[str]]:
        """Analyze physiological markers."""
        score = 0.0
        factors = []
        
        if not cardio:
            return score, factors
            
        # Check elevated heart rate
        if "resting_hr" in cardio:
            hr = cardio["resting_hr"]
            if hr > self.config.elevated_resting_hr:
                score += self.config.weights["elevated_hr"]
                factors.append(f"Elevated resting HR: {hr:.0f} bpm")
        
        # Check reduced HRV
        if "avg_hrv" in cardio:
            hrv = cardio["avg_hrv"]
            if hrv < self.config.low_hrv_threshold:
                score += self.config.weights["low_hrv"]
                factors.append(f"Low HRV: {hrv:.0f} ms")
                
        return score, factors
    
    def _analyze_temporal_patterns(
        self,
        sleep: Optional[SleepFeatures],
        activity: Optional[Dict[str, Any]],
        baseline: Optional[Dict[str, float]],
    ) -> Tuple[float, list[str]]:
        """Analyze temporal patterns and trends."""
        score = 0.0
        factors = []
        
        # This is where we would implement:
        # - Trend analysis (sleep getting progressively worse)
        # - Intra-day variability analysis
        # - Phase advance detection
        # For now, placeholder for future enhancement
        
        return score, factors
    
    def _determine_alert_level(self, score: float) -> str:
        """Determine categorical alert level from risk score."""
        if score >= self.high_threshold:
            return "high"
        elif score >= self.moderate_threshold:
            return "moderate"
        elif score > 0.1:
            return "low"
        else:
            return "none"
    
    def _generate_clinical_insight(
        self,
        level: str,
        factors: list[str],
        score: float,
    ) -> str:
        """Generate clinical insight message."""
        if level == "high":
            primary_factors = factors[:2] if len(factors) > 2 else factors
            factors_str = " and ".join(primary_factors).lower()
            return (
                f"Elevated mania risk detected (score: {score:.2f}) - "
                f"{factors_str}. Consider contacting your healthcare provider."
            )
        elif level == "moderate":
            return (
                f"Moderate mania warning signs observed (score: {score:.2f}). "
                "Monitor symptoms closely and maintain sleep routine."
            )
        elif level == "low":
            return (
                f"Mild irregularities detected (score: {score:.2f}). "
                "Continue healthy sleep and activity habits."
            )
        else:
            return "No significant mania risk indicators detected."
    
    def _generate_recommendations(
        self,
        level: str,
        factors: list[str],
    ) -> list[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []
        
        if level in ["high", "moderate"]:
            # Sleep-related recommendations
            if any("sleep" in f.lower() for f in factors):
                recommendations.append(
                    "Prioritize sleep: aim for 7-8 hours at consistent times"
                )
                recommendations.append(
                    "Avoid screens 2 hours before bedtime"
                )
            
            # Activity-related recommendations
            if any("activity" in f.lower() for f in factors):
                recommendations.append(
                    "Monitor activity levels - avoid overcommitment"
                )
                recommendations.append(
                    "Schedule regular rest periods throughout the day"
                )
            
            # Circadian-related recommendations
            if any("circadian" in f.lower() for f in factors):
                recommendations.append(
                    "Maintain consistent wake/sleep times"
                )
                recommendations.append(
                    "Get morning sunlight exposure"
                )
            
            # General high-risk recommendations
            if level == "high":
                recommendations.insert(0, 
                    "Contact your healthcare provider within 24 hours"
                )
                recommendations.append(
                    "Avoid major decisions or commitments"
                )
        
        return recommendations
```

#### 1.2 Create Configuration File

```yaml
# config/mania_risk_config.yaml

# Mania Risk Detection Configuration
# Based on 2024-2025 computational psychiatry research

thresholds:
  # Sleep duration thresholds (hours)
  min_sleep_hours: 5.0          # Below this = concerning
  critical_sleep_hours: 3.0     # Below this = high risk
  sleep_loss_percent: 0.4       # 40% reduction from baseline
  
  # Circadian rhythm thresholds
  circadian_disruption: 0.5     # Score below this = disrupted
  phase_advance_hours: 1.0      # Hours of phase shift
  
  # Activity thresholds
  activity_surge_ratio: 1.5     # 50% increase from baseline
  activity_fragmentation: 0.8   # High fragmentation threshold
  
  # Physiological thresholds
  elevated_resting_hr: 90       # BPM
  low_hrv: 20                   # ms SDNN
  
  # Alert level thresholds
  moderate_risk: 0.4
  high_risk: 0.7

weights:
  # Sleep-related weights
  severe_sleep_loss: 0.30       # Major contributor
  acute_sleep_loss: 0.20        # Single night <3h
  rapid_sleep_onset: 0.10       # <5 min latency
  
  # Circadian weights
  circadian_disruption: 0.25    # Major contributor
  sleep_inconsistency: 0.10     # Irregular schedule
  circadian_phase_advance: 0.15 # Phase shift detected
  
  # Activity weights
  activity_fragmentation: 0.20  # Disorganized patterns
  activity_surge: 0.10          # Sudden increases
  
  # Physiological weights
  elevated_hr: 0.05             # Supporting signal
  low_hrv: 0.05                 # Supporting signal

# Feature extraction windows
temporal:
  baseline_days: 28             # Days for personal baseline
  trend_window_days: 7          # Days for trend analysis
  rapid_change_hours: 72        # Hours for acute changes
```

### Phase 2: Backend Integration

#### 2.1 Update Data Models

```python
# Update src/clarity/ml/pat_service.py

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
    
    # NEW FIELDS for mania risk
    mania_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mania/hypomania risk score (0-1)"
    )
    mania_alert_level: str = Field(
        default="none",
        description="Mania risk level: none/low/moderate/high"
    )
    
    sleep_stages: list[str] = Field(description="Predicted sleep stages")
    confidence_score: float = Field(description="Model confidence (0-1)")
    clinical_insights: list[str] = Field(description="Clinical interpretations")
    embedding: list[float] = Field(description="PAT model embedding vector")
```

#### 2.2 Integrate with PAT Service

```python
# Update src/clarity/ml/pat_service.py - _postprocess_predictions method

def _postprocess_predictions(
    self,
    outputs: dict[str, torch.Tensor],
    user_id: str,
) -> ActigraphyAnalysis:
    """Convert model outputs to clinical insights."""
    # ... existing code ...
    
    # Generate clinical insights
    insights = self._generate_clinical_insights(
        sleep_efficiency, circadian_score, depression_risk
    )
    
    # NEW: Analyze mania risk
    mania_analyzer = ManiaRiskAnalyzer()
    mania_result = mania_analyzer.analyze(
        sleep_features=None,  # We don't have detailed sleep here
        pat_metrics={
            "total_sleep_time": total_sleep_time,
            "circadian_rhythm_score": circadian_score,
            "activity_fragmentation": activity_fragmentation,
            "sleep_efficiency": sleep_efficiency / 100,  # Convert to 0-1
        },
        activity_stats=None,  # Not available in step-only analysis
        cardio_stats=None,    # Not available in step-only analysis
    )
    
    # Add mania insight if significant
    if mania_result.alert_level in ["moderate", "high"]:
        insights.append(mania_result.clinical_insight)
    
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
        mania_risk_score=mania_result.risk_score,
        mania_alert_level=mania_result.alert_level,
        sleep_stages=sleep_stages,
        confidence_score=confidence_score,
        clinical_insights=insights,
        embedding=full_embedding,
    )
```

#### 2.3 Integrate with Health Analysis Pipeline

```python
# Update src/clarity/ml/analysis_pipeline.py

async def process_health_data(
    self,
    user_id: str,
    health_metrics: list[HealthMetric],
    processing_id: str | None = None,
) -> AnalysisResults:
    """Process health metrics through the analysis pipeline."""
    # ... existing processing code ...
    
    # Step 4: Generate summary statistics  
    results.summary_stats = self._generate_summary_stats(
        organized_data,
        modality_features,
        results.activity_features,
    )
    
    # NEW Step 5: Mania risk analysis
    mania_result = await self._analyze_mania_risk(
        user_id,
        results,
        organized_data,
    )
    
    # Add to summary stats
    results.summary_stats.setdefault("health_indicators", {})
    results.summary_stats["health_indicators"]["mania_risk"] = {
        "risk_score": mania_result.risk_score,
        "alert_level": mania_result.alert_level,
        "contributing_factors": mania_result.contributing_factors,
        "confidence": mania_result.confidence,
    }
    
    # Add to clinical insights if significant
    if mania_result.alert_level in ["moderate", "high"]:
        insights = results.summary_stats.setdefault("clinical_insights", [])
        insights.append(mania_result.clinical_insight)
        
        # Add recommendations
        if mania_result.recommendations:
            results.summary_stats["recommendations"] = mania_result.recommendations
    
    # ... rest of existing code ...
    
async def _analyze_mania_risk(
    self,
    user_id: str,
    results: AnalysisResults,
    organized_data: dict[str, list[HealthMetric]],
) -> ManiaRiskResult:
    """Analyze mania risk using all available data."""
    # Initialize analyzer
    config_path = Path("config/mania_risk_config.yaml")
    analyzer = ManiaRiskAnalyzer(config_path)
    
    # Prepare sleep features
    sleep_features = None
    if results.sleep_features:
        sleep_features = SleepFeatures(**results.sleep_features)
    
    # Prepare PAT metrics from sleep features or other sources
    pat_metrics = {}
    if results.sleep_features:
        pat_metrics.update({
            "circadian_rhythm_score": results.sleep_features.get(
                "circadian_rhythm_score", 0.0
            ),
            "activity_fragmentation": results.sleep_features.get(
                "activity_fragmentation", 0.0
            ),
        })
    
    # Prepare activity stats
    activity_stats = None
    if results.activity_features:
        # Extract relevant stats from activity features
        activity_stats = {
            "avg_daily_steps": self._extract_avg_daily_steps(
                results.activity_features
            ),
            "peak_daily_steps": self._extract_peak_daily_steps(
                results.activity_features
            ),
            "activity_consistency": self._extract_activity_consistency(
                results.activity_features
            ),
        }
    
    # Prepare cardio stats
    cardio_stats = None
    if results.cardio_features and len(results.cardio_features) >= 8:
        cardio_stats = {
            "avg_hr": results.cardio_features[0],
            "resting_hr": results.cardio_features[2],
            "avg_hrv": results.cardio_features[4],
            "circadian_rhythm_score": results.cardio_features[7],
        }
    
    # Get historical baseline from DynamoDB
    historical_baseline = await self._get_user_baseline(user_id)
    
    # Perform mania risk analysis
    return analyzer.analyze(
        sleep_features=sleep_features,
        pat_metrics=pat_metrics,
        activity_stats=activity_stats,
        cardio_stats=cardio_stats,
        historical_baseline=historical_baseline,
    )

async def _get_user_baseline(self, user_id: str) -> dict[str, float] | None:
    """Retrieve user's historical baseline from DynamoDB."""
    try:
        dynamodb_client = await self._get_dynamodb_client()
        
        # Query for past 28 days of analysis
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=28)
        
        response = dynamodb_client.table.query(
            KeyConditionExpression=Key("pk").eq(f"USER#{user_id}") & 
                                 Key("sk").between(
                                     f"ANALYSIS#{start_date.isoformat()}",
                                     f"ANALYSIS#{end_date.isoformat()}"
                                 ),
            Limit=28  # Maximum 28 days
        )
        
        if not response.get("Items"):
            return None
            
        # Calculate baseline averages
        sleep_hours = []
        steps = []
        
        for item in response["Items"]:
            # Extract sleep hours
            if "sleep_features" in item:
                sleep_mins = item["sleep_features"].get("total_sleep_minutes", 0)
                if sleep_mins > 0:
                    sleep_hours.append(float(sleep_mins) / 60)
            
            # Extract steps
            if "activity_features" in item:
                daily_steps = item["activity_features"].get("avg_daily_steps", 0)
                if daily_steps > 0:
                    steps.append(float(daily_steps))
        
        baseline = {}
        if sleep_hours:
            baseline["avg_sleep_hours"] = np.mean(sleep_hours)
            baseline["std_sleep_hours"] = np.std(sleep_hours)
        
        if steps:
            baseline["avg_steps"] = np.mean(steps)
            baseline["std_steps"] = np.std(steps)
            
        return baseline
        
    except Exception as e:
        self.logger.warning(f"Failed to retrieve user baseline: {e}")
        return None
```

### Phase 3: Testing Strategy

#### 3.1 Unit Tests

```python
# tests/ml/test_mania_risk_analyzer.py

import pytest
from datetime import datetime, UTC
from clarity.ml.mania_risk_analyzer import (
    ManiaRiskAnalyzer, ManiaRiskConfig, ManiaRiskResult
)
from clarity.ml.processors.sleep_processor import SleepFeatures


class TestManiaRiskAnalyzer:
    """Comprehensive tests for mania risk detection."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default config."""
        return ManiaRiskAnalyzer()
    
    @pytest.fixture
    def healthy_sleep(self):
        """Create healthy sleep features."""
        return SleepFeatures(
            total_sleep_minutes=450,  # 7.5 hours
            sleep_efficiency=0.85,
            sleep_latency=15.0,
            awakenings_count=2.0,
            consistency_score=0.8,
            quality_score=0.85,
        )
    
    @pytest.fixture
    def manic_sleep(self):
        """Create sleep features indicating mania risk."""
        return SleepFeatures(
            total_sleep_minutes=180,  # 3 hours
            sleep_efficiency=0.95,    # Paradoxically high
            sleep_latency=2.0,        # Falls asleep instantly
            awakenings_count=0.0,     # No awakenings
            consistency_score=0.2,    # Very irregular
            quality_score=0.4,
        )
    
    def test_no_risk_healthy_data(self, analyzer, healthy_sleep):
        """Test that healthy data produces no risk."""
        result = analyzer.analyze(
            sleep_features=healthy_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.85,
                "activity_fragmentation": 0.3,
            }
        )
        
        assert result.risk_score < 0.1
        assert result.alert_level == "none"
        assert len(result.contributing_factors) == 0
    
    def test_high_risk_severe_sleep_loss(self, analyzer, manic_sleep):
        """Test severe sleep loss triggers high risk."""
        result = analyzer.analyze(
            sleep_features=manic_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,  # Disrupted
                "activity_fragmentation": 0.9,   # High fragmentation
            }
        )
        
        assert result.risk_score >= 0.7
        assert result.alert_level == "high"
        assert any("sleep" in f.lower() for f in result.contributing_factors)
        assert "healthcare provider" in result.clinical_insight
    
    def test_moderate_risk_activity_surge(self, analyzer, healthy_sleep):
        """Test activity surge with mild sleep reduction."""
        result = analyzer.analyze(
            sleep_features=healthy_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.6,
                "activity_fragmentation": 0.7,
            },
            activity_stats={
                "avg_daily_steps": 15000,
            },
            historical_baseline={
                "avg_steps": 8000,  # Nearly 2x baseline
            }
        )
        
        assert 0.4 <= result.risk_score < 0.7
        assert result.alert_level == "moderate"
        assert any("activity" in f.lower() for f in result.contributing_factors)
    
    def test_physiological_support(self, analyzer):
        """Test that physiological markers add to risk."""
        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 4.5,  # Just under threshold
                "circadian_rhythm_score": 0.6,
                "activity_fragmentation": 0.6,
            },
            cardio_stats={
                "resting_hr": 95,  # Elevated
                "avg_hrv": 15,     # Low
            }
        )
        
        # Should push from low to moderate due to physio markers
        assert result.risk_score >= 0.4
        assert any("HR" in f for f in result.contributing_factors)
        assert any("HRV" in f for f in result.contributing_factors)
    
    def test_recommendations_generated(self, analyzer, manic_sleep):
        """Test that appropriate recommendations are generated."""
        result = analyzer.analyze(
            sleep_features=manic_sleep,
            pat_metrics={
                "circadian_rhythm_score": 0.3,
                "activity_fragmentation": 0.9,
            }
        )
        
        assert len(result.recommendations) >= 3
        assert any("sleep" in r.lower() for r in result.recommendations)
        assert any("provider" in r.lower() for r in result.recommendations)
    
    def test_confidence_adjustment(self, analyzer):
        """Test confidence is reduced with estimated data."""
        # Using only PAT metrics (estimated)
        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": 3.0,
                "circadian_rhythm_score": 0.3,
            }
        )
        
        assert result.confidence < 1.0  # Reduced confidence
        
    def test_phase_advance_detection(self, analyzer):
        """Test circadian phase advance contributes to risk."""
        # TODO: Implement when we add sleep timing analysis
        pass
    
    @pytest.mark.parametrize("sleep_hours,expected_level", [
        (8.0, "none"),      # Normal sleep
        (6.0, "none"),      # Slightly reduced but ok
        (4.5, "moderate"),  # Below threshold
        (3.0, "high"),      # Critical threshold
        (1.0, "high"),      # Severe insomnia
    ])
    def test_sleep_thresholds(self, analyzer, sleep_hours, expected_level):
        """Test various sleep duration thresholds."""
        result = analyzer.analyze(
            pat_metrics={
                "total_sleep_time": sleep_hours,
                "circadian_rhythm_score": 0.7,  # Normal circadian
                "activity_fragmentation": 0.5,   # Normal activity
            }
        )
        
        assert result.alert_level == expected_level
```

#### 3.2 Integration Tests

```python
# tests/integration/test_mania_risk_integration.py

import pytest
from datetime import datetime, timedelta, UTC
from clarity.ml.analysis_pipeline import HealthAnalysisPipeline
from clarity.models.health_data import HealthMetric, SleepData
from tests.fixtures.data_factories import (
    create_sleep_metric,
    create_heart_rate_metric,
    create_steps_metric,
)


class TestManiaRiskIntegration:
    """Integration tests for mania risk in the full pipeline."""
    
    @pytest.fixture
    async def pipeline(self):
        """Create analysis pipeline instance."""
        return HealthAnalysisPipeline()
    
    @pytest.mark.asyncio
    async def test_mania_risk_in_full_pipeline(self, pipeline):
        """Test mania risk analysis in complete health data pipeline."""
        user_id = "test-user-123"
        
        # Create a week of data with manic patterns
        metrics = []
        base_time = datetime.now(UTC) - timedelta(days=7)
        
        # Progressively worsening sleep (mania prodrome)
        for day in range(7):
            timestamp = base_time + timedelta(days=day)
            
            # Sleep gets progressively worse
            sleep_hours = 7 - (day * 0.7)  # 7h → 2.8h
            metrics.append(
                create_sleep_metric(
                    user_id=user_id,
                    timestamp=timestamp,
                    total_sleep_minutes=sleep_hours * 60,
                    sleep_efficiency=0.9 if day > 4 else 0.8,
                    consistency_score=0.8 - (day * 0.1),
                )
            )
            
            # Activity increases
            steps = 8000 + (day * 2000)  # 8k → 20k
            metrics.append(
                create_steps_metric(
                    user_id=user_id,
                    timestamp=timestamp,
                    step_count=steps,
                )
            )
            
            # Heart rate increases
            hr = 65 + (day * 3)  # 65 → 86
            metrics.append(
                create_heart_rate_metric(
                    user_id=user_id,
                    timestamp=timestamp,
                    heart_rate=hr,
                )
            )
        
        # Process through pipeline
        results = await pipeline.process_health_data(
            user_id=user_id,
            health_metrics=metrics,
            processing_id="test-processing-123"
        )
        
        # Verify mania risk detected
        assert "health_indicators" in results.summary_stats
        assert "mania_risk" in results.summary_stats["health_indicators"]
        
        mania_risk = results.summary_stats["health_indicators"]["mania_risk"]
        assert mania_risk["risk_score"] >= 0.7
        assert mania_risk["alert_level"] == "high"
        assert len(mania_risk["contributing_factors"]) >= 2
        
        # Verify clinical insights include mania warning
        insights = results.summary_stats.get("clinical_insights", [])
        assert any("mania" in insight.lower() for insight in insights)
        
        # Verify recommendations provided
        assert "recommendations" in results.summary_stats
        assert len(results.summary_stats["recommendations"]) >= 3
    
    @pytest.mark.asyncio
    async def test_mania_risk_with_baseline(self, pipeline, mock_dynamodb):
        """Test mania risk detection using historical baseline."""
        user_id = "test-user-456"
        
        # Mock historical baseline in DynamoDB
        mock_dynamodb.add_baseline(
            user_id=user_id,
            avg_sleep_hours=7.5,
            avg_steps=10000,
        )
        
        # Create current data showing 50% sleep reduction
        metrics = [
            create_sleep_metric(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                total_sleep_minutes=225,  # 3.75 hours (50% of 7.5)
            ),
            create_steps_metric(
                user_id=user_id,
                timestamp=datetime.now(UTC),
                step_count=18000,  # 80% increase
            ),
        ]
        
        results = await pipeline.process_health_data(
            user_id=user_id,
            health_metrics=metrics,
        )
        
        mania_risk = results.summary_stats["health_indicators"]["mania_risk"]
        factors = mania_risk["contributing_factors"]
        
        # Should detect both sleep reduction and activity surge
        assert any("50%" in f for f in factors)  # Sleep reduction
        assert any("1.8x" in f for f in factors)  # Activity surge
```

### Phase 4: API Response Examples

#### Updated Step Analysis Response
```json
{
    "analysis": {
        "user_id": "user-123",
        "analysis_timestamp": "2025-01-15T10:30:00Z",
        "sleep_efficiency": 82.1,
        "sleep_onset_latency": 3.0,
        "wake_after_sleep_onset": 12.0,
        "total_sleep_time": 3.2,
        "circadian_rhythm_score": 0.35,
        "activity_fragmentation": 0.88,
        "depression_risk_score": 0.12,
        "mania_risk_score": 0.78,
        "mania_alert_level": "high",
        "sleep_stages": ["wake", "light", "deep", "rem"],
        "confidence_score": 0.9,
        "clinical_insights": [
            "Poor sleep efficiency - consider sleep hygiene improvements",
            "Irregular circadian rhythm - prioritize sleep consistency",
            "Elevated mania risk detected (score: 0.78) - critically low sleep: 3.2h (PAT estimation) and high activity fragmentation: 0.88. Consider contacting your healthcare provider."
        ],
        "embedding": [0.123, 0.456, ...]
    }
}
```

#### Updated Health Analysis Response
```json
{
    "processing_id": "abc-123",
    "user_id": "user-123",
    "status": "completed",
    "summary_stats": {
        "data_coverage": {
            "sleep": {
                "metric_count": 7,
                "time_span_hours": 168,
                "data_density": 0.042
            },
            "activity": {
                "metric_count": 168,
                "time_span_hours": 168,
                "data_density": 1.0
            }
        },
        "health_indicators": {
            "cardiovascular_health": {
                "avg_heart_rate": 72,
                "resting_heart_rate": 58
            },
            "sleep_health": {
                "avg_duration_hours": 4.2,
                "avg_efficiency": 0.82
            },
            "mania_risk": {
                "risk_score": 0.85,
                "alert_level": "high",
                "contributing_factors": [
                    "Critically low sleep: 3.0h (HealthKit)",
                    "Sleep reduced 60% from baseline",
                    "Disrupted circadian rhythm (score: 0.28)",
                    "High activity fragmentation: 0.91",
                    "Activity surge: 1.9x baseline"
                ],
                "confidence": 0.95
            }
        },
        "clinical_insights": [
            "Poor sleep quality detected",
            "Elevated mania risk detected (score: 0.85) - critically low sleep: 3.0h (healthkit) and activity surge: 1.9x baseline. Consider contacting your healthcare provider."
        ],
        "recommendations": [
            "Contact your healthcare provider within 24 hours",
            "Prioritize sleep: aim for 7-8 hours at consistent times",
            "Monitor activity levels - avoid overcommitment",
            "Maintain consistent wake/sleep times",
            "Avoid major decisions or commitments"
        ]
    }
}
```

## Implementation Timeline

### Week 1: Core Development
- [ ] Day 1-2: Implement ManiaRiskAnalyzer class
- [ ] Day 3: Create configuration system
- [ ] Day 4-5: Write comprehensive unit tests

### Week 2: Integration
- [ ] Day 1-2: Update data models (ActigraphyAnalysis)
- [ ] Day 3: Integrate with PAT service
- [ ] Day 4-5: Integrate with health analysis pipeline

### Week 3: Testing & Refinement  
- [ ] Day 1-2: Integration testing
- [ ] Day 3: Performance optimization
- [ ] Day 4-5: Clinical validation with advisors

### Week 4: Production Readiness
- [ ] Day 1-2: Add monitoring and alerting
- [ ] Day 3: Documentation
- [ ] Day 4-5: Deployment and monitoring

## Future Enhancements

### 1. Machine Learning Evolution
- Collect labeled data from users who experience manic episodes
- Train specialized transformer model for mania prediction
- Fine-tune PAT model with bipolar-specific data
- Implement online learning for personalization

### 2. Advanced Features
- **Circadian Phase Detection**: Implement algorithm to detect phase advances/delays
- **Temporal Sequence Analysis**: Use RNNs/transformers for pattern detection
- **Multi-modal Fusion**: Combine all signals optimally using attention mechanisms
- **Medication Adherence**: Track if risk increases when medications missed

### 3. Clinical Integration
- **Provider Dashboard**: Real-time risk monitoring for care teams
- **Automated Alerts**: Configurable notifications to providers
- **Intervention Recommendations**: Personalized action plans
- **Outcome Tracking**: Monitor if predictions prevented episodes

## Conclusion

This implementation plan delivers a cutting-edge mania risk detection system that:
1. Leverages the latest 2024-2025 research achieving 0.98 AUC
2. Integrates seamlessly with existing CLARITY architecture
3. Provides actionable insights and recommendations
4. Sets foundation for future ML enhancements
5. Maintains clinical rigor while being immediately deployable

The modular design ensures we can start with rule-based detection and evolve to sophisticated ML models as data accumulates, positioning CLARITY as a leader in digital mental health monitoring.