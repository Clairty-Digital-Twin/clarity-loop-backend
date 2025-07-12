# CLARITY Digital Twin: Bipolar Monitoring Enhancement Plan
## Based on 2024-2025 State-of-the-Art Research

### Executive Summary
This plan details the implementation of advanced bipolar disorder (BD) monitoring capabilities based on recent breakthroughs in computational psychiatry. The enhancements will achieve:
- **98% AUC for mania prediction** (Lim et al., 2024)
- **7-day advance warning for depression** (Ortiz et al., 2025)
- **89% accuracy using activity variability** (Oura Ring study, 2025)

### Key Research Findings to Implement

#### 1. **Circadian Phase Shifts** (Lim et al., 2024 - npj Digital Medicine)
- **Finding**: Daily circadian timing shifts are the STRONGEST predictors
- **Specifics**:
  - Phase advances (earlier sleep) → mania (AUC 0.98)
  - Phase delays (later sleep) → depression (AUC 0.80)
  - Just sleep-wake data needed - no complex sensors
- **Implementation**: `CircadianPhaseDetector` module

#### 2. **Activity Variability Windows** (Ortiz et al., 2025)
- **Finding**: Variability gives earliest warning signals
- **Specifics**:
  - Day-to-day step count variability → 7 days warning for depression
  - 12-hour sleep pattern variability → 87% accuracy for hypomania
  - Within-night sleep stage variability → 3 days warning
- **Implementation**: `VariabilityAnalyzer` module

#### 3. **Sleep as Primary Signal** (Multiple studies)
- **Finding**: 75% of BD patients report sleep problems before mania
- **Specifics**:
  - Sleep reduction + phase advance = strong mania predictor
  - Sleep increase + phase delay = depression predictor
  - Both duration AND timing matter
- **Implementation**: Enhanced `ManiaRiskAnalyzer`

#### 4. **Personalized Baselines** (Lipschitz et al., 2025)
- **Finding**: Individual baselines beat population models
- **Specifics**:
  - BiMM forest achieved 86% AUC for depression, 85% for mania
  - Personalization is key - one person's active is another's sedentary
- **Implementation**: `PersonalBaselineTracker`

### Architecture Components

#### 1. Core Analysis Pipeline Enhancement
```
HealthKit Data → Processors → Feature Extraction → Risk Analysis → Alerts
                     ↓              ↓                    ↓
                SleepProcessor  CircadianPhase    ManiaRiskAnalyzer
                ActivityProc    VariabilityCalc   DepressionAnalyzer
                CardioProc      PhaseShiftDetect  MixedStateDetector
```

#### 2. New Modules to Implement

##### A. `CircadianPhaseDetector`
- **Purpose**: Detect phase advances/delays that predict episodes
- **Inputs**: Sleep start/end times from HealthKit
- **Outputs**: 
  - Phase shift magnitude (hours)
  - Direction (advance/delay/stable)
  - Clinical significance score
- **Algorithm**:
  ```python
  1. Calculate sleep midpoint for each night
  2. Compare to personal baseline (median of past 14-28 days)
  3. Detect shifts > 1 hour (advance) or > 1.5 hours (delay)
  4. Weight by consistency and magnitude
  ```

##### B. `VariabilityAnalyzer`
- **Purpose**: Calculate activity/sleep variability metrics
- **Inputs**: Time series data (steps, sleep, HR)
- **Outputs**:
  - Coefficient of variation for different windows
  - Spike detection results
  - Trend direction (increasing/stable/decreasing)
- **Algorithm**:
  ```python
  1. Calculate rolling std dev for 12h, 24h, 3d, 7d windows
  2. Implement time-frequency spike detection (Ortiz method)
  3. Compare to personal baseline variability
  4. Flag significant increases (>1.5x baseline)
  ```

##### C. `PersonalBaselineTracker`
- **Purpose**: Maintain individual baselines for all metrics
- **Inputs**: Historical data (28+ days preferred)
- **Outputs**:
  - Personal averages, medians, percentiles
  - Circadian rhythm profile
  - Normal variability ranges
- **Storage**: DynamoDB with efficient updates

##### D. `EpisodePredictionEngine`
- **Purpose**: Combine all signals for episode prediction
- **Inputs**: All analyzer outputs
- **Outputs**:
  - Episode type prediction (mania/depression/mixed/stable)
  - Confidence score
  - Time horizon (days until likely episode)
- **Algorithm**: Weighted ensemble of all signals

#### 3. Enhanced Risk Scoring

##### Current Weights (to update):
```yaml
severe_sleep_loss: 0.45
circadian_phase_advance: 0.35  # INCREASE based on Lim et al.
activity_variability: 0.30     # ADD based on Ortiz et al.
phase_delay: 0.25             # ADD for depression
rapid_sleep_onset: 0.10
```

##### New Scientific Weights:
```yaml
# Mania Predictors (based on AUCs)
circadian_phase_advance: 0.40   # Strongest predictor
severe_sleep_loss: 0.30
activity_surge: 0.15
sleep_variability_increase: 0.15

# Depression Predictors
circadian_phase_delay: 0.35
activity_variability_increase: 0.35  # 7-day warning
sleep_increase: 0.20
social_withdrawal: 0.10
```

### Implementation Plan

#### Phase 1: Core Infrastructure (Week 1)
1. **CircadianPhaseDetector**
   - [ ] Implement phase shift calculation
   - [ ] Add wraparound handling (23:00 → 01:00)
   - [ ] Create baseline comparison logic
   - [ ] Add confidence scoring

2. **VariabilityAnalyzer**
   - [ ] Implement rolling window calculations
   - [ ] Add spike detection algorithm
   - [ ] Create multi-timescale analysis (12h, 24h, 3d, 7d)
   - [ ] Build trend detection

3. **PersonalBaselineTracker**
   - [ ] Design DynamoDB schema
   - [ ] Implement efficient update algorithms
   - [ ] Add data quality checks
   - [ ] Create fallback for new users

#### Phase 2: Integration (Week 2)
1. **Update ManiaRiskAnalyzer**
   - [ ] Integrate CircadianPhaseDetector
   - [ ] Add variability signals
   - [ ] Update weights based on research
   - [ ] Enhance temporal pattern analysis

2. **Create DepressionRiskAnalyzer**
   - [ ] Mirror structure of ManiaRiskAnalyzer
   - [ ] Focus on phase delays and variability
   - [ ] Add 7-day prediction window
   - [ ] Include activity withdrawal detection

3. **Implement MixedStateDetector**
   - [ ] Detect conflicting signals
   - [ ] Handle simultaneous mania/depression markers
   - [ ] Add uncertainty quantification

#### Phase 3: Advanced Features (Week 3)
1. **Time-to-Event Prediction**
   - [ ] Implement survival analysis concepts
   - [ ] Estimate days until episode
   - [ ] Add confidence intervals

2. **Personalization Engine**
   - [ ] Learn individual patterns
   - [ ] Adjust thresholds per user
   - [ ] Handle sparse data gracefully

3. **Alert Optimization**
   - [ ] Implement smart alert fatigue prevention
   - [ ] Add contextual recommendations
   - [ ] Create provider dashboard views

### Data Requirements

#### From Apple HealthKit:
- **Sleep**: Start/end times, stages (if available)
- **Activity**: Steps, distance, active energy
- **Heart**: HR, HRV
- **Workouts**: Type, duration, intensity

#### From PAT Model:
- Circadian rhythm score
- Activity fragmentation
- Sleep quality metrics
- 96-dim embeddings

#### Computed Features:
- Sleep midpoint (hours from midnight)
- Phase shifts (vs baseline)
- Variability metrics (multiple windows)
- Trend indicators

### Validation Strategy

1. **Unit Tests**
   - Each module tested independently
   - Edge cases (wraparound, missing data)
   - Performance under load

2. **Integration Tests**
   - Full pipeline testing
   - Multiple user scenarios
   - Alert generation verification

3. **Clinical Validation**
   - Compare to published AUCs
   - Sensitivity/specificity analysis
   - False positive rate monitoring

### Performance Targets

Based on literature:
- Mania detection: >95% AUC (Lim: 98%)
- Depression detection: >80% AUC (Lim: 80%)
- Hypomania detection: >85% accuracy (Ortiz: 87%)
- Advance warning: 3-7 days
- False positive rate: <10%

### Risk Mitigation

1. **Data Quality**
   - Handle missing data gracefully
   - Require minimum data density
   - Validate sensor readings

2. **Clinical Safety**
   - Never diagnose, only flag risks
   - Always recommend provider consultation
   - Include confidence scores

3. **Privacy**
   - All analysis on-device when possible
   - Minimal data retention
   - HIPAA-compliant logging

### Success Metrics

1. **Technical**
   - All tests passing (>95% coverage)
   - Response time <500ms
   - 99.9% uptime

2. **Clinical**
   - Match published AUCs within 5%
   - User-reported episode detection >80%
   - Provider acceptance >90%

3. **User Experience**
   - Alert acknowledgment >70%
   - Feature usage >60% MAU
   - NPS >50

### Timeline

- **Week 1**: Core modules implementation
- **Week 2**: Integration and testing
- **Week 3**: Advanced features and optimization
- **Week 4**: Validation and documentation
- **Week 5**: Provider dashboard and deployment

### Next Steps

1. Review and approve plan
2. Set up development environment
3. Begin CircadianPhaseDetector implementation
4. Create test data generators
5. Schedule clinical advisor review

### References

1. Lim et al. (2024). "Predicting mood episodes with sleep data." npj Digital Medicine.
2. Ortiz et al. (2025). "Activity variability in bipolar disorder." Journal of Affective Disorders.
3. Lipschitz et al. (2025). "Fitbit-based mood prediction." International Journal of Bipolar Disorders.
4. Anmella et al. (2024). "TIMEBASE protocol." BMC Psychiatry.

---

*This plan integrates cutting-edge research to create a world-class bipolar monitoring system that will genuinely help patients and providers catch episodes early.*