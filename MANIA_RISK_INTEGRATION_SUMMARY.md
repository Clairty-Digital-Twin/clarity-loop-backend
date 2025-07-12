# Mania Risk Analyzer Integration Summary

## What Was Implemented

Successfully integrated the ManiaRiskAnalyzer into the main health analysis pipeline for comprehensive bipolar disorder risk detection.

### Key Changes Made

1. **Updated `src/clarity/ml/analysis_pipeline.py`**:
   - Added imports for ManiaRiskAnalyzer and required types
   - Added `_analyze_mania_risk()` method that:
     - Creates ManiaRiskAnalyzer with config file
     - Prepares data from all sources (sleep, activity, cardio)
     - Retrieves historical baseline from DynamoDB
     - Returns ManiaRiskResult
   - Added `_get_user_baseline()` to query DynamoDB for 28-day baseline
   - Integrated mania analysis after Step 4 in `process_health_data()`
   - Added mania risk to summary_stats under health_indicators
   - Added recommendations to summary_stats when risk is moderate/high

2. **Helper Methods Added**:
   - `_extract_avg_daily_steps()`: Extract average daily steps from activity features
   - `_extract_peak_daily_steps()`: Extract peak daily steps
   - `_extract_activity_consistency()`: Extract activity consistency score

3. **Integration Points**:
   - Mania risk analysis runs after all modality processing
   - Uses data from:
     - SleepProcessor (sleep features)
     - ActivityProcessor (step counts, activity metrics)
     - CardioProcessor (heart rate, HRV, circadian rhythm)
     - PAT model (estimated sleep, activity fragmentation)
   - Stores results in DynamoDB along with other analysis data

4. **Test Coverage**:
   - Created comprehensive integration tests in `test_analysis_pipeline_mania_integration.py`
   - Tests cover:
     - High risk detection with sleep deprivation and activity surge
     - Risk detection without historical baseline
     - No risk with healthy data
     - Recommendations generation for high risk
     - PAT-only data analysis

### API Response Structure

The mania risk analysis adds the following to the health analysis response:

```json
{
  "summary_stats": {
    "health_indicators": {
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
      "Elevated mania risk detected (score: 0.85) - critically low sleep: 3.0h (HealthKit) and activity surge: 1.9x baseline. Consider contacting your healthcare provider."
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

### Next Steps

1. Monitor real-world performance and adjust thresholds based on user feedback
2. Collect labeled data for future ML model training
3. Add circadian phase advance detection when sleep timing data available
4. Implement temporal trend analysis for progressive pattern detection

The integration is complete and all tests are passing. The mania risk analyzer now runs automatically as part of the health data analysis pipeline.