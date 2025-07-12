# Clinical Reality Check - Bipolar Monitoring

## What We're Actually Building

A **RULE-BASED SCREENING TOOL** that flags potential risk indicators based on established clinical patterns. This is NOT:
- A diagnostic tool
- An ML prediction system  
- A replacement for clinical care
- FDA-approved or clinically validated

## Real Clinical Rules We Can Implement

### 1. Sleep Reduction (DSM-5 Criterion A2)
- **Rule**: Sleep < 5 hours/night for 3+ consecutive nights → Flag HIGH risk
- **Basis**: Decreased need for sleep is a core manic symptom
- **Implementation**: Simple threshold on sleep_minutes

### 2. Sleep Phase Advance  
- **Rule**: Sleep midpoint shifts earlier by >2 hours → Flag MODERATE risk
- **Basis**: Clinical observation, not quantified accuracy
- **Implementation**: Compare weekly average sleep midpoints

### 3. Increased Activity (DSM-5 Criterion B1)
- **Rule**: Steps >150% of personal baseline → Flag MODERATE risk
- **Basis**: Psychomotor agitation/goal-directed activity
- **Implementation**: Compare to 4-week rolling average

### 4. Sleep Variability
- **Rule**: Sleep duration SD >2 hours over 7 days → Flag LOW risk
- **Basis**: Circadian disruption precedes episodes
- **Implementation**: Rolling standard deviation

### 5. Reduced Heart Rate Variability
- **Rule**: HRV <20ms sustained → Flag LOW risk  
- **Basis**: Autonomic dysfunction in mood episodes
- **Implementation**: Simple threshold

## What We CANNOT Claim

1. **"98% AUC"** - Research papers achieved this with:
   - Labeled episode data (we don't have)
   - Trained ML models (we're not using)
   - Clinical validation (we haven't done)

2. **"7-day advance warning"** - This requires:
   - Individual episode history
   - Validated prediction models
   - Longitudinal validation

3. **"Clinical-grade"** - Reserved for:
   - FDA-approved devices
   - Clinically validated algorithms
   - Peer-reviewed accuracy studies

## Honest Implementation Path

### Phase 1: Rule-Based Screening (Current)
- Implement simple threshold rules
- Track basic metrics (sleep, steps, HR, HRV)
- Generate risk flags, NOT predictions
- Document all assumptions

### Phase 2: Data Collection (Future)
- Gather user-reported mood states
- Build labeled dataset
- Track rule performance

### Phase 3: ML Development (Future)
- Train personalized models
- Validate against clinical outcomes
- Publish accuracy metrics
- Seek regulatory approval

## Code Guidelines

1. **Naming**: Use "risk_indicator" not "predictor"
2. **Outputs**: Return "screening_score" not "diagnosis"
3. **Docs**: State "heuristic rules" not "AI/ML"
4. **Weights**: Label as "arbitrary" until validated

## Example Honest Documentation

```python
def calculate_mania_risk_score(metrics):
    """
    Calculate heuristic risk score for mania screening.
    
    This is a RULE-BASED system using simple thresholds inspired
    by DSM-5 criteria. It has NOT been clinically validated.
    
    Returns:
        score: 0-1 risk indicator (NOT a probability)
        factors: List of triggered rules
        
    WARNING: This is a screening tool only. Consult healthcare
    providers for actual diagnosis and treatment.
    """
```

## The Bottom Line

We're building a useful screening tool based on established clinical patterns. It can help users and clinicians track concerning changes. But we must be honest about its limitations and avoid ML/AI hype until we have the data and validation to back it up.