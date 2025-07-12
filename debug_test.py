from clarity.ml.mania_risk_analyzer import ManiaRiskAnalyzer
from clarity.ml.processors.sleep_processor import SleepFeatures

analyzer = ManiaRiskAnalyzer()

# Healthy sleep: 7.5 hours, good efficiency
healthy_sleep = SleepFeatures(
    total_sleep_minutes=450,  # 7.5 hours
    sleep_efficiency=0.85,
    sleep_latency=15.0,
    awakenings_count=2.0,
    consistency_score=0.8,
    quality_score=0.85,
)

print(f"Sleep object: {healthy_sleep}")
print(f"Sleep total_sleep_minutes: {healthy_sleep.total_sleep_minutes}")
print(f"Sleep hours: {healthy_sleep.total_sleep_minutes / 60}")

# Test _analyze_sleep directly
sleep_score, sleep_factors, sleep_conf = analyzer._analyze_sleep(
    healthy_sleep, 
    {"circadian_rhythm_score": 0.85, "activity_fragmentation": 0.3},
    None
)
print(f"\nDirect _analyze_sleep result:")
print(f"Score: {sleep_score}, Factors: {sleep_factors}, Confidence: {sleep_conf}")

result = analyzer.analyze(
    sleep_features=healthy_sleep,
    pat_metrics={
        "circadian_rhythm_score": 0.85,
        "activity_fragmentation": 0.3,
    }
)

print(f"\nFull analysis result:")
print(f"Risk score: {result.risk_score}")
print(f"Alert level: {result.alert_level}")
print(f"Contributing factors: {result.contributing_factors}")
print(f"Clinical insight: {result.clinical_insight}")