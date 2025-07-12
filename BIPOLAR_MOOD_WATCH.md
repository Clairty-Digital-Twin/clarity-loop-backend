Mania Risk Forecasting Module – Ultra-Enhanced Design

1. Mania Risk Module – Research-Informed Rule Engine

Goal: Implement a new backend component that computes a mania risk score and alert level from Apple Watch data, using clinically grounded rules (no ML needed for v1). This module exploits known early-warning biomarkers of manic/hypomanic episodes, aligning with state-of-the-art computational psychiatry research to deliver a “shock the tech world” capability even without proprietary training data.

Key Features & Inputs: The mania risk analyzer will ingest the same processed signals our pipeline already derives (sleep, activity, circadian, etc.), focusing on patterns known to precede manic episodes:
	•	Severe Sleep Reduction: A dramatically decreased need for sleep is a hallmark prodrome of mania. The module will flag if average sleep duration (from SleepProcessor) falls below a critical threshold (e.g. <5 hours/night) or if there’s an abrupt drop (>40% decrease) from the user’s normal sleep. For instance, if the user normally sleeps ~7 hours but over the last few days has been getting 3-4 hours, that’s a strong risk indicator. We use SleepFeatures.total_sleep_minutes (weekly mean) and also look at night-to-night variability: if even one night was extremely short (say 0–3 hours), we boost the risk score significantly. This captures acute sleep loss events often seen just before mania onset. We’ll make thresholds configurable (e.g. default 4.5 hours mean or a single night <3h triggers high risk).
	•	Circadian Rhythm Disruption: Internal clock shifts and irregular daily routines are tightly linked to mood episodes. Our PAT model already outputs a circadian_rhythm_score (0–1) indicating how regular the 24-hour activity pattern is. A low score (e.g. <0.5) means the user’s cycle is erratic – e.g. sleeping/waking at odd times or no stable day-night pattern – which often foreshadows mania ￼. The mania module will add substantial weight to risk if circadian_score is low or has sharply changed. We also utilize sleep timing from SleepFeatures.consistency_score (regularity of bedtime): if the user’s sleep onset times vary widely (consistency_score below ~0.4), it reinforces circadian disruption. Research has shown that circadian phase shifts (e.g. a phase advance, where the body clock shifts earlier) are specifically associated with impending manic episodes. For example, if a user suddenly starts waking up 3–4 hours earlier than usual full of energy, that’s a red flag.
	•	Activity Level Escalation & Fragmentation: Increased psychomotor activity – restlessness, pacing, overscheduling – is a core feature of mania. We capture this via two signals:
	•	Activity fragmentation: The PAT model infers an activity_fragmentation index (0–1) reflecting how disorganized or broken-up the activity pattern is. A high fragmentation (e.g. >0.8) means lots of spurts of activity and rest at irregular intervals, which correlates with manic energy surges (the person is “constantly on the go” in spurts). We raise mania risk proportionally to fragmentation.
	•	Overall activity surge: Using ActivityProcessor outputs (e.g. total daily steps, exercise minutes from HealthKit), we detect if the user is far more active than their recent baseline. For instance, if average daily steps this week are 50%+ higher than the prior week, or if there were days with extreme exercise spikes, we increment the risk. While exercise is generally good, a sudden unexplained spike in activity level (especially coupled with little sleep) can indicate emerging hypomania. This aligns with clinical lore and is backed by studies finding that “increased activity, more than elevated mood, is the core feature of mania”. Our rule might say: if current-week steps > 1.5× the past-week steps (and >10k), add moderate risk.
	•	Physiological Arousal: Mania often comes with heightened autonomic arousal – e.g. elevated resting heart rate and lower heart rate variability (HRV) as the body is in a revved-up state. Our CardioProcessor yields features including resting_hr and avg_hrv. The mania module will treat extreme values as supportive signals: e.g. if resting_hr is in the 90th+ percentile for age (or say >90 bpm sustained), or if HRV (SDNN) is very low (<20 ms), we add a small bump to risk. These alone won’t trigger an alert (since high heart rate could be exercise), but if they coincide with the above behavioral signs, they strengthen confidence. We also check the day-night heart rate difference (from Cardio’s circadian_rhythm_score on HR): a healthy pattern is lower HR at night. If that difference blurs (i.e. similar HR day and night), it suggests the user’s sympathetic nervous system is constantly “on” – which could happen in mania. We factor that in if available.
	•	Rapid Changes & Trends: Importantly, the module doesn’t just look at absolute values but also short-term changes over time. We incorporate temporal sequencing by analyzing the past week (or two) of data in sub-intervals:
	•	We compare the last 2–3 days vs. the earlier days of the week to spot an accelerating trend (e.g. sleep dropping each day, activity climbing). If, say, the user slept 7h, 6h, 5h, then 4h on consecutive nights, that downward spiral triggers a higher risk than just a low weekly average. Similarly, if their activity fragmentation was mild early in the week but became very high in the last couple of days, we respond to that spike.
	•	Inspired by recent research, we pay special attention to intra-day variability. One study found that changes in 12-hour sleep/activity variability were the earliest and strongest indicators of an oncoming mood episode, giving up to ~3 days of advance warning. In practice, this means we look for erratic swings within the day: e.g. a day where the user took an unusually long midday nap or pulled an all-nighter followed by hyperactive morning. Our rule-based approach can’t do full frequency analysis, but we approximate by flagging any day with huge deviation from the norm (for sleep or activity). For example, a day split into two sleep bouts or a day with an abnormal burst of activity at midnight would lower the circadian score and raise fragmentation – which our logic already catches. Essentially, fine-grained disruptions count extra, since mania can escalate quickly within 24–72 hours. By recalculating risk each day (every time new HealthKit data arrives), our system will catch these rapid changes in near-real-time.

Scoring Algorithm: We will implement this as a weighted rule-based score on a 0 to 1 scale. Each feature above contributes points to a cumulative mania_risk_score. For transparency and easy tuning, we’ll define weight constants (or read from a config file, e.g. mania_rules.yaml) for each condition. For example:
	•	Sleep average <5h: +0.30 (major contributor)
	•	Any single night <3h: +0.20 (additional, or even auto-high if two nights zero sleep)
	•	Sleep latency extremely short (<5 min) in context of low total sleep: +0.1 (the user falls asleep instantly due to exhaustion, which paradoxically can occur during manic buildup after long insomnia bouts)
	•	Circadian rhythm score <0.5: +0.25
	•	Bedtime inconsistency (sleep consistency_score <0.4): +0.1
	•	Activity fragmentation >0.8: +0.2
	•	Huge activity surge (steps or active minutes >1.5× baseline): +0.1
	•	Resting HR very high or HRV very low: +0.1 (combined cap, so physiology at most contributes 0.1–0.2)

After summing, we clamp the score to [0,1]. Then we derive a categorical alert level:
	•	mania_alert_level = "high" if score ≥ 0.7 (suggesting strong mania signals present),
	•	"moderate" if score between ~0.4–0.7 (some warning signs),
	•	"none" (or "low") if below 0.4 (no significant indications).

These thresholds (MODERATE_MANIA_RISK = 0.4, HIGH_MANIA_RISK = 0.7 by default) mirror how the system handles depression risk (which uses 0.4/0.7 cutoffs in PAT). They can be adjusted as we calibrate the system with real data or clinician input.

Output: The module returns two numbers (mania_risk_score and mania_alert_level) and one insight string. The insight message provides a concise, user-friendly interpretation to append to the clinical_insights. We’ll craft messages similar in tone to existing ones:
	•	If high risk: “Elevated mania risk – patterns of severely reduced sleep and irregular activity detected; consider reaching out to your care team.” This aligns with our evidence-based alerting while remaining a suggestion, not a diagnosis.
	•	If moderate: “Moderate signs of hypomania – some sleep and activity irregularities observed; monitor closely.”
	•	If none/low: We may omit adding an insight (to avoid clutter) or we can include a reassuring note like “No mania-related patterns observed.” (In practice, we might skip a “none” insight unless product wants an explicit all-clear.)

These messages are generated within the mania module so they are immediately available to the API response. They will reference the specific patterns detected (e.g. if it was mainly sleep loss and fragmentation, the message can mention those). In code, the ManiaRiskAnalyzer.analyze() method could assemble the message based on which rules fired (for example, if sleep was the dominant factor vs. circadian disruption). For now a generic message per level is fine, but the framework allows tailoring it further.

Why Rule-Based: Since we currently lack a large bipolar patient dataset, a deterministic approach is appropriate and safe. By encoding clinically validated heuristics, we provide value from day one. This design is also “ML-ready” – we isolate feature computation in one place, so that in future we can swap the internal logic with a learned model. For example, the analyzer can first gather all features into a vector (sleep hrs, variability, circadian score, HR, etc.) and if use_ml_model is enabled (and a model is loaded) then output = model.predict(vector). The output would still map to the same risk score and level. This way, once we do have training data (perhaps from users who log mood or from literature simulations), we can train a classifier or even fine-tune our Fusion Transformer to predict mania risk probability ￼ (noting that one study achieved AUC 0.95–0.98 in predicting manic episodes using similar sleep/circadian features). Our rule-based v1 sets the stage for that “god-mode” ML upgrade without requiring any API changes.

2. Backend Integration & Data Flow

We will integrate the mania risk computation seamlessly into our existing FastAPI pipeline, ensuring the data flows and outputs are extended but not disrupted:

2.1 Code Organization: Create a new file src/clarity/ml/mania_model.py (or mania_analyzer.py) for this module. It will contain:
	•	A ManiaRiskAnalyzer class with an analyze(...) method as described.
	•	Configuration constants or a loader for thresholds (we might load a YAML into a dict mania_rules on module import).
	•	Optionally, a Pydantic model ManiaRiskResult to formalize the output (or we can just return a tuple/dict).

For now, pseudocode inside ManiaRiskAnalyzer.analyze might look like:

class ManiaRiskAnalyzer:
    def __init__(self):
        # Load thresholds and weights, e.g.:
        cfg = load_yaml("mania_rules.yaml")
        self.thresholds = cfg["thresholds"]
        self.weights = cfg["weights"]

    def analyze(self, sleep: SleepFeatures | None, pat_metrics: dict[str, float], 
                activity_stats: dict[str, float] | None = None, 
                cardio: dict[str, float] | None = None) -> tuple[float, str, str]:
        score = 0.0
        # 1. Sleep duration check
        if sleep:
            hours = sleep.total_sleep_minutes / 60
        else:
            hours = pat_metrics.get("total_sleep_time", 0.0)
        baseline = self.thresholds["min_sleep_hours"]  # e.g. 5
        if hours < baseline:
            score += self.weights["sleep_loss"]         # e.g. +0.3
        # Check single-night minima if available (we might need to pass recent nights' data)
        if sleep and sleep.total_sleep_minutes:  # if multiple nights were aggregated
            # Suppose we somehow have access to min or std of nightly durations:
            if sleep.min_sleep_minutes and sleep.min_sleep_minutes/60 < 3:
                score += self.weights["acute_sleep_loss"]  # e.g. +0.2

        # 2. Sleep latency (fast sleep onset)
        if sleep and sleep.sleep_latency < 5:
            score += self.weights["short_latency"]        # e.g. +0.1

        # 3. Circadian rhythm
        circ_score = pat_metrics.get("circadian_rhythm_score")
        if circ_score is not None and circ_score < self.thresholds["circadian_score"]:
            score += self.weights["circadian_disruption"] # e.g. +0.25
        # Sleep consistency (bedtime variability)
        if sleep and sleep.consistency_score < 0.4:
            score += self.weights["irregular_bedtime"]    # e.g. +0.1

        # 4. Activity fragmentation
        frag = pat_metrics.get("activity_fragmentation")
        if frag is not None and frag > self.thresholds["activity_frag"]:
            score += self.weights["fragmentation"]        # e.g. +0.2

        # 5. Activity surge
        if activity_stats:
            # Compare current vs previous steps if provided
            if activity_stats.get("steps_ratio_vs_baseline", 0) > 1.5:
                score += self.weights["activity_surge"]   # e.g. +0.1

        # 6. Cardio signals
        if cardio:
            if cardio.get("resting_hr", 0) > 90:
                score += self.weights["high_resting_hr"]  # e.g. +0.05
            if cardio.get("avg_hrv", 0) < 20:
                score += self.weights["low_hrv"]          # e.g. +0.05

        # Clamp score to [0,1]
        score = max(0.0, min(score, 1.0))
        # Determine level
        if score >= self.thresholds["high"]:
            level = "high"
        elif score >= self.thresholds["moderate"]:
            level = "moderate"
        else:
            level = "none"

        # Generate insight message
        insight = ""
        if level == "high":
            insight = "Elevated mania risk – patterns of greatly reduced sleep and irregular activity detected; consider contacting your provider."
        elif level == "moderate":
            insight = "Moderate signs of hypomania – some sleep and activity irregularities observed; please monitor."
        # (if level none, we might return empty string or a neutral message)

        return score, level, insight

(Above is illustrative – the actual implementation will refine how we get min sleep, baseline, etc. For example, if we have access to historical analysis results, we might compute steps_ratio_vs_baseline by comparing this week’s average steps to an earlier period stored in DynamoDB. If not, we rely on general thresholds.)

2.2 Extending Data Models: We will augment the output schemas to include mania risk:
	•	ActigraphyAnalysis model: In src/clarity/ml/pat_service.py, the Pydantic class ActigraphyAnalysis (which structures the PAT model results) will get two new fields:

class ActigraphyAnalysis(BaseModel):
    user_id: str
    analysis_timestamp: str
    sleep_efficiency: float
    sleep_onset_latency: float
    wake_after_sleep_onset: float
    total_sleep_time: float
    circadian_rhythm_score: float
    activity_fragmentation: float
    depression_risk_score: float
    mania_risk_score: float = Field(description="Mania risk score (0-1)")
    mania_alert_level: str = Field(description="Mania risk level (none/moderate/high)")
    sleep_stages: list[str]
    confidence_score: float
    clinical_insights: list[str]
    embedding: list[float]

We insert the mania fields right after depression_risk_score for logical grouping. This change means any API responses or stored analysis objects will now carry these fields (with minimal overhead, as they’re just one float and one short string).

	•	Summary stats / DynamoDB: Our pipeline currently stores a JSON of results.summary_stats and other metrics in DynamoDB. We will include mania info there as well so it’s queryable for trends:
	•	Add results.summary_stats["mania_risk_score"] and ["mania_alert_level"]. Alternatively, we might group it under a "mental_health" section of health_indicators. For simplicity, we can top-level it in summary_stats or inside an existing key. Since depression risk wasn’t explicitly in summary_stats (it was only in ActigraphyAnalysis), we might just include mania in the insights text and as part of ActigraphyAnalysis. However, having it in summary_stats could help if we want to easily retrieve the latest risk level without re-running PAT.
	•	When saving to Dynamo, our code in analysis_pipeline.process_health_data will convert floats to Decimal and put the new fields into the item. DynamoDB being schemaless means we don’t need to migrate, just start storing the new attributes. (We will ensure any retrieval code doesn’t break on additional fields – typically it won’t.)

2.3 Pipeline Integration: We will integrate the mania analysis at two points:
	•	(A) Real-time Step Data Analysis (FastAPI endpoint): When a user calls the POST /step-analysis endpoint with their recent step counts (which we convert to actigraphy for PAT), the PAT model returns an ActigraphyAnalysis object. We will update the PAT service to attach mania risk before returning it. Concretely, in PATModelService._postprocess_predictions (in pat_service.py), after assembling the ActigraphyAnalysis with all standard fields, we call our analyzer:

analysis = ActigraphyAnalysis(..., depression_risk_score=depression_risk, ...,
                              clinical_insights=insights, embedding=full_embedding)
# Now inject mania risk
mania_analyzer = ManiaRiskAnalyzer()
m_score, m_level, m_msg = mania_analyzer.analyze(
    sleep=None,  # no direct SleepFeatures since we only have PAT’s derived metrics
    pat_metrics={
        "total_sleep_time": analysis.total_sleep_time,
        "circadian_rhythm_score": analysis.circadian_rhythm_score,
        "activity_fragmentation": analysis.activity_fragmentation
    },
    activity_stats=None,
    cardio=None
)
analysis.mania_risk_score = m_score
analysis.mania_alert_level = m_level
if m_msg:
    analysis.clinical_insights.append(m_msg)

We pass in PAT’s own estimates of sleep and circadian metrics as a proxy. For example, PAT’s total_sleep_time (in hours) comes from the transformer’s sleep head. While not as exact as HealthKit sleep data, it gives us a basis for mania assessment in this context. The mania analyzer sees, for instance, total_sleep_time=4.2h and circadian_score=0.3 from PAT, and will correctly flag high risk. This ensures that even in step-only mode, where we have no direct sleep logs, the system can raise a warning – a unique capability showcasing the power of our integrated approach. The updated ActigraphyAnalysis (with mania fields populated) is then returned through the API. The client/mobile app will receive JSON like:

"analysis": {
   "user_id": "...", "analysis_timestamp": "...",
   "sleep_efficiency": 82.1, "sleep_onset_latency": 14.0, "wake_after_sleep_onset": 34.0,
   "total_sleep_time": 6.5, "circadian_rhythm_score": 0.45, "activity_fragmentation": 0.81,
   "depression_risk_score": 0.12,
   "mania_risk_score": 0.75,
   "mania_alert_level": "high",
   "sleep_stages": [...], "confidence_score": 0.9,
   "clinical_insights": [
       "Poor sleep efficiency - consider sleep hygiene improvements",
       "Irregular circadian rhythm - prioritize sleep consistency",
       "Elevated depression risk indicators - consider professional consultation",
       "**Elevated mania risk – patterns of severely reduced sleep and irregular activity detected; consider contacting provider.**"
   ],
   "embedding": [ ... 96 values ... ]
}

Note: We might mark up the mania insight with some highlighting (as above) when displaying, or prepend “⚠️” to emphasize it. The insight generation can include such symbols to draw attention if desired.

	•	(B) Full Health Analysis Pipeline: When the backend processes a batch of HealthKit data (sleep, HR, steps, etc.) via HealthAnalysisPipeline.process_health_data, we’ll hook in mania risk computation before finalizing results. In analysis_pipeline.py, after Step 4 (summary stat generation) and before saving to DB, we add:

# Step 5: Mania risk analysis (new)
mania_analyzer = ManiaRiskAnalyzer()
m_score, m_level, m_msg = mania_analyzer.analyze(
    sleep = SleepFeatures(**results.sleep_features) if results.sleep_features else None,
    pat_metrics = {
        "circadian_rhythm_score": results.sleep_features.get("circadian_rhythm_score", 0.0) 
                                   if results.sleep_features else 0.0,
        "activity_fragmentation": results.sleep_features.get("activity_fragmentation", 0.0) 
                                   if results.sleep_features else 0.0
    },
    activity_stats = self._derive_activity_stats(results.activity_features),
    cardio = self._derive_cardio_stats(results.cardio_features)
)
results.summary_stats.setdefault("health_indicators", {})
results.summary_stats["health_indicators"]["mania_risk_score"] = float(m_score)
results.summary_stats["health_indicators"]["mania_alert_level"] = m_level
if m_msg:
    # Append to clinical insights if summary_stats carries them, or directly to results (if we choose to store insights there)
    insights_list = results.summary_stats.setdefault("clinical_insights", [])
    insights_list.append(m_msg)

Here we leverage actual SleepFeatures from HealthKit, which should be more accurate than PAT’s inferred sleep. We also illustrate using helper methods _derive_activity_stats to compute things like steps baseline ratio (e.g. comparing the current period’s steps to historical average from DB or simply marking if >10k steps/day average), and _derive_cardio_stats to pull resting_hr, avg_hrv, etc., from the cardio_features vector (knowing the order from CardioProcessor). These helpers make it easy to pass data into the analyzer in the expected format.
The result is that for each analysis record saved, the DynamoDB item will include something like "mania_risk_score": Decimal('0.75'), "mania_alert_level": "high" in the summary_stats.health_indicators. This not only returns to the user (if they fetch analysis results via an API) but also allows us to track population trends – e.g. we could query how many users hit high mania risk in a given month, etc.

2.4 Handling Time and State: Because mania risk is highly temporal, we considered how to incorporate longer-term context:
	•	The pipeline typically runs on a rolling 7-day window of data (as indicated by ActigraphyInput.duration_hours = 168 ￼). Within that window, our analyzer looks at patterns as described (e.g. differences between days). If the user continuously uploads data, this becomes a moving window updated daily. Thus, the mania risk score will naturally respond to week-over-week changes. To further incorporate history, we can use the stored baseline from DynamoDB. For example, we might compute a user’s 28-day average sleep or steps and use that as the baseline to detect unusual deviations. In v1, if such baseline is readily available (perhaps from an earlier analysis item), we will include it. If not, our rules use healthy-population expectations (e.g. <5h sleep is universally low).
	•	We ensure time alignment: The mania analyzer will be invoked after we’ve aggregated the week’s data. If data is sparse or missing days, we’ll interpret carefully (e.g. if only 3 nights of data in the week, a low average might just mean incomplete data – our insights code can check results.data_coverage to avoid false alarms).
	•	Additionally, if mania_alert_level is “high”, we might set a flag in the DB or trigger a notification workflow. While not in scope for this coding task, we note that the architecture has an Alerts concept (e.g. anomalies). We can piggyback on that: e.g. if high risk, add an item to an alerts table or send an SNS message. This ensures the system not only computes the risk but also acts on it (the “🚨 Health Alerts” in our architecture ￼ ￼ now includes mood episode alerts).

2.5 No Impact on Existing Functionality: By adding fields (non-breaking) and appending to insights, we preserve backward compatibility. Clients unaware of mania data will just ignore the extra fields. Our tests will verify that if mania_analyzer is off (e.g. if no sleep data, it should safely output none/0). The pipeline and PAT service remain robust even if mania analysis fails (we’ll wrap it in a try/except so any error doesn’t crash the whole analysis; at worst we log an error and proceed without mania fields).

In summary, the backend changes ensure that whether a user triggers analysis via the streaming step data endpoint or the asynchronous health data upload, the output now includes a cutting-edge mania risk assessment. This feature operates entirely on the data we already collect (steps, heart rate, sleep from HealthKit), effectively transforming our app into an early warning system for bipolar manic episodes using only passive data – a capability that closely matches recent research breakthroughs and sets our digital health platform apart.

3. Validation, Testing, and Future Evolution

Unit Tests: We will develop comprehensive tests for this module:
	•	Rule correctness: Feed synthetic data into ManiaRiskAnalyzer and assert expected outcomes. For example, construct a SleepFeatures with 4h avg sleep, low consistency, etc., and verify mania_alert_level == "high". Test boundary conditions around thresholds (e.g. exactly 5h sleep = moderate vs 4.9h triggers high risk). We’ll also test combinations: e.g. sleep just under threshold but circadian normal should maybe yield moderate, whereas combined with low circadian = high.
	•	Temporal scenarios: Simulate a time series of user data over days. We can create a small function to compute a moving mania score: e.g. Day1 normal, Day2 slightly less sleep, Day3 very little sleep + high activity. The test will feed the cumulative 3-day data and ensure the risk score by Day3 is high and that the insight message corresponds to sleep loss. This checks that our logic effectively “catches” an emerging trend.
	•	Integration with pipeline: If possible in tests, run the pipeline on sample HealthMetric lists that mimic real uploads. For instance, craft a week of HealthMetrics where the last two days have drastic changes. After process_health_data, confirm that results.summary_stats["health_indicators"]["mania_alert_level"] is set to the correct value and that clinical_insights contains the new message. We’ll also test that if the data is perfectly healthy (8h sleep, regular routine), the mania_risk_score comes out 0 and no alert is added.
	•	API contract: Update API tests (e.g. test_step_analysis_endpoint) to ensure the response now includes mania fields. We’ll verify that the values are within [0,1] and the level is a known string. If needed, add a schema validation that mania_risk_score is always present (even if 0) so that front-end doesn’t encounter missing fields.

Clinical Validation: Though we can’t fully validate without real patient data, we will conduct expert review of the rule set. We’ll present the thresholds and weights to our clinical advisors (e.g. psychiatrists specialized in bipolar disorder) to get feedback. The design already leans on published research (e.g. Ortiz 2025 for variability, Dartmouth/KAIST 2024 for circadian shifts, etc.), which we’ll cite in an internal brief to build confidence. We expect iteration on the exact cutoffs. For instance, if 5 hours seems too low (some patients regularly sleep only 5-6h when well), we might raise it or incorporate personalization (like using the user’s own average as baseline).

Monitoring in Production: We will add logging at key points:
	•	When mania risk is computed, log an info: "ManiaRisk computed: user X, score=0.75, level=high (flags: low_sleep, irregular_circadian)". Including which flags triggered can help later analysis. (We’ll gather those flags during scoring – e.g. keep a list of triggered rule names).
	•	If level == "high", also log a warning or emit a CloudWatch metric ManiaRiskHigh with value 1. This allows ops to set up an alarm if, say, a sudden spike in high-risk cases occurs (which could indicate either something in the world, or a bug if false positives).
	•	We will periodically query DynamoDB for the distribution of mania_risk_score to see if our thresholds need tuning (e.g. if almost everyone is “moderate”, we may need to adjust sensitivity to avoid alarm fatigue).

Next Steps (Beyond v1): Our rule-based approach can be seen as a baseline model for mania detection. As data accumulates, we plan to enhance it in two major ways:
	1.	Personalized Baselines: The system can learn each user’s typical sleep and activity patterns (after a few weeks of data). Then mania risk can be computed relative to individual baselines rather than fixed thresholds. For example, if a certain user normally sleeps only 5.5h (short sleeper phenotype), our static rule might always flag them; but a personalized model would recognize 5.5h is normal for them and only trigger if they drop to, say, 3h. We could implement this by storing long-term averages and standard deviations per user in DynamoDB or in the user profile.
	2.	Machine Learning Model: With labeled outcomes (e.g. if users report mood or if we collaborate with research datasets), we could train a classifier (like a logistic regression or small neural net) using our extracted features. Given the literature, a model might achieve very high accuracy – e.g. predicting manic episodes with AUC ~0.95 using just passive data. We could even fine-tune our multi-modal Fusion Transformer on bipolar patients: feeding it the 7-day time series and having it output a risk score directly. The architecture already supports multi-modal inputs (sleep, HR, activity) ￼ ￼; we would simply add a new “Mania Risk Head” to the transformer that learns from sequence patterns (perhaps using a training strategy similar to anomaly detection). That’s a v2 “god mode” idea. For now, our rule engine is an interpretable, deterministic stand-in that still brings cutting-edge capability to the platform.

By integrating all the above, CLARITY becomes one of the first digital health platforms to offer an automated mania early warning system based on wearables. This not only has huge clinical value (preventing hospitalizations by early intervention) but also distinguishes our backend technologically. We are effectively operationalizing insights from recent bipolar research in real-time for users – something even big tech has not deployed yet. Our v1 mania risk module, though rule-based, is built on rigorous scientific evidence and is designed to be both immediately impactful and future-proof for ML. This will be a headline feature of our product, demonstrating how our backend’s flexible architecture can rapidly incorporate the latest computational psychiatry breakthroughs into a practical, scalable service.

Sources: We ground this design in peer-reviewed findings. For example, Lim et al. (2024) showed that using just sleep-wake data from wearables, one can predict next-day manic episodes with ~98% accuracy, underscoring the feasibility of our approach. Ortiz et al. (2025) found that heightened variability in sleep and activity often appears around 3 days before a bipolar mood episode, which directly informs our focus on rapid changes. Moreover, a large-scale causal analysis by Lee et al. (2024) confirmed that circadian disruptions causally precede mood symptoms in bipolar I disorder, reinforcing why our module prioritizes circadian metrics. By leveraging these insights (reduced sleep, circadian phase shifts, hyperactivity) and implementing them in our pipeline, we deliver a “gorgeous” v1 feature that is scientifically robust and practically attainable with our current backend and data. This is how we turn our CLARITY digital twin into a proactive guardian against manic episodes, breaking new ground in wearable-based mental health tech. ￼