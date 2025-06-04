# State-of-the-Art Sleep Staging Transformers (2024–2025) and Integration Plan

## Cutting-Edge Sleep Staging Models (2024–2025)

Recent research has produced several transformer-based models for automated sleep staging using standard datasets (Sleep-EDF, SHHS/MESA from NSRR, etc.). Below we highlight notable models and their characteristics:

* **PFTSleep (2024)** – *“Patch-based Foundation Transformer for Sleep”*. This is a self-supervised transformer model that encodes an **entire night (8-hour) multi-channel PSG** at 125 Hz across 7 signals (EEG, EOG, EMG, ECG, respiration, SpO₂, etc.). After pretraining via masked autoregression (based on the PatchTST architecture), PFTSleep achieves **state-of-the-art sleep stage classification** performance. On the SHHS validation set it reached AUROC of 0.95–0.99 for each stage (Wake, N1, N2, N3, REM), outperforming prior methods with notable gains in sensitivity and F1-score. *Production notes:* This model is a *foundation model* requiring full-night signals – powerful but heavy. As of 2024 it’s a research prototype (medRxiv preprint) without an official open-source implementation, so direct deployment would require custom reimplementation or cooperation with the authors.

* **SleepVST (CVPR 2024)** – *“Sleep Video Spectral Transformer”*. A transformer leveraging pretraining on contact sensor data (cardio-respiratory signals from SHHS/MESA) to enable **camera-based sleep staging** from near-infrared video. SleepVST achieved **state-of-the-art** results on wearables: Cohen’s κ of 0.75–0.77 on SHHS/MESA (5-class staging). Notably, when transfer-learning to a 100% contact-free video dataset, it produced a **four-class sleep staging** (Wake, Light/Core, Deep, REM) accuracy of **78.8% (κ=0.71)**. This 4-class scheme aligns well with Apple’s categorization (Core/Light, Deep, REM, Wake). *Production notes:* The approach is novel (CVPR ’24); code may be forthcoming, but it’s complex (involves video signal extraction and two-stage pretraining). It shows that **transformers can map to Apple’s sleep categories** after appropriate training, but an off-the-shelf open implementation is not yet available.

* **SleepTransformer (Phan *et al.*, IEEE TBME 2022)** – An earlier transformer model using a **sequence-to-sequence** approach for sleep staging. It operates on single-channel EEG (converting 30s epochs to time–frequency “images”) and employs self-attention at both epoch and sequence level for interpretability. SleepTransformer achieved **81.4% accuracy** on the Sleep-EDF 2018 dataset without any pretraining – competitive with CNN/RNN methods of the time. It also introduced uncertainty quantification in its predictions. *Production notes:* This model’s **code is open-sourced** in TensorFlow 1.x (with pre-trained weights on SHHS). However, the aging tech stack (TF1, MATLAB scripts for prep) and moderate performance mean it may need updates for modern production. Still, its availability and clinical focus (interpretability, uncertainty) make it a useful reference design.

* *(Additional)* **SleepViT / SleepXViT (2023)** – These are Vision Transformer-based models on spectral or multimodal inputs. For example, SleepViTransformer (Peng *et al.*, 2023) uses patch-wise spectrogram tokens to classify epochs, and *FlexSleepTransformer* (Guo *et al.*, 2024) introduces flexible channel configurations for multi-dataset training. These report performance on Sleep-EDF (\~82–85% accuracy) and show transformers’ versatility in sleep staging. Most are research code (often MATLAB/Python prototypes) rather than turnkey solutions.

**Bottom line:** *While cutting-edge transformer models for sleep staging exist, none comes as a ready-made, production-hardened package.* PFTSleep leads in accuracy (thanks to huge data and full-night context), but it’s not yet open-source. SleepTransformer is open, but uses older frameworks. No off-the-shelf model explicitly outputs Apple’s “Core/Deep/REM/Wake” classes – though models like SleepVST demonstrate that mapping standard 5-stage outputs to those 4 categories is straightforward (e.g. merge N1/N2 into “Core/Light”, use N3 as “Deep”). Given the lack of a plug-and-play solution that is both **state-of-the-art and production-ready**, the most pragmatic approach is to **integrate a custom SleepProcessor into our existing backend**.

## Integration Plan: **SleepProcessor** Microservice in the FastAPI Backend

In our clean-architecture backend (FastAPI, Pydantic models, modular ML pipeline), we will add a dedicated **SleepProcessor** component to handle sleep staging data – much like the existing PAT actigraphy module, but for *actual* sleep stage inputs. The goal is to ingest Apple HealthKit sleep records (stages and summaries), produce standardized metrics (Core/REM/Deep/Wake breakdown, totals, etc.), and feed these into our insight generation flow (Gemini 2.5 LLM). Key integration steps:

### **1. Module & File Layout**

Create a new processor module, e.g. **`src/clarity/ml/processors/sleep_processor.py`**, alongside `activity_processor.py` and others. This module will define a class `SleepProcessor` with a similar interface:

* `SleepProcessor.process(self, metrics: list[HealthMetric]) -> list[dict[str, Any]]`: Accepts a list of `HealthMetric` objects of type SLEEP\_ANALYSIS and returns extracted features/key metrics. We follow the pattern of ActivityProcessor – processing all sleep metrics (possibly multiple nights) in one call.

Inside `SleepProcessor`, we implement logic to compute **nightly summary features** from the raw sleep data. Likely features (each as a dict with `"feature_name"` and `"value"`) include:

* **total\_sleep\_time** (hours per night, and perhaps average if multiple nights),
* **sleep\_efficiency** (ratio or % – computed as total sleep time / time in bed, if in-bed duration available),
* **sleep\_latency** (time to fall asleep, a.k.a. sleep onset latency),
* **wake\_after\_sleep\_onset** (WASO – minutes awake after initially falling asleep, sum of awake periods during the night),
* **awakenings\_count** (number of distinct wake periods – “disturbances”),
* **sleep\_stage\_minutes** (minutes in Core/Light, Deep, REM – we can output percentages or keep raw minutes for each stage),
* **consistency\_score** (optional – a 0–1 score reflecting **sleep consistency** across nights, e.g. based on variability in sleep duration or bedtimes over the week).

For example, if three nights of data are provided, `SleepProcessor` might compute each night’s metrics and then an overall average or consistency metric. The output could be a list like:

```json
[ 
  {"feature_name": "average_sleep_duration", "value": 7.2},
  {"feature_name": "sleep_efficiency", "value": 0.88},
  {"feature_name": "average_wake_episodes", "value": 2},
  {"feature_name": "rem_percentage", "value": 20.5},
  {"feature_name": "deep_percentage", "value": 15.0},
  {"feature_name": "sleep_consistency_score", "value": 0.9}
]
```

*(The exact feature set can be refined; the aim is to expose key sleep stats for insights.)*

We will log the processor initialization (similar to ActivityProcessor’s logger info) and handle exceptions robustly (ensuring a failure in sleep analysis doesn’t crash the pipeline, returning an error dict if needed).

### **2. Pydantic Model Extensions**

Our Pydantic data models already include a `SleepData` schema under `HealthData` (for example, `clarity.models.health_data.SleepData`) which captures comprehensive sleep info. Notably, `SleepData` has fields for:

* `total_sleep_minutes`,
* `sleep_efficiency` (as 0–1 float),
* `time_to_sleep_minutes` (sleep latency),
* `wake_count` (number of awakenings),
* `sleep_stages: dict[SleepStage, int]` (minutes in each stage),
* `sleep_start` and `sleep_end` timestamps.

These correspond well to HealthKit’s native sleep fields. For instance, HealthKit provides categorical sleep samples (`HKCategoryTypeIdentifierSleepAnalysis`) with values like “Core”, “Deep”, “REM”, etc., and we map those to our `SleepStage` enum (`LIGHT` for Core, `DEEP` for Deep, `REM` for REM, and `AWAKE`). We also ingest `HKQuantityTypeIdentifierTimeInBed` as a separate metric if available, which essentially corresponds to the difference between `sleep_start` and `sleep_end`.

**Schema adjustments:** We will ensure that our Pydantic validators handle Apple’s data:

* When HealthKit data is uploaded, our ingestion logic (likely in the HealthDataService or Preprocessor) should populate a `HealthMetric` with `metric_type = SLEEP_ANALYSIS` and a filled `SleepData` sub-model. Indeed, in the current code, the HealthKit integration already creates `HealthMetricType.SLEEP_ANALYSIS` entries with SleepData (using provided or default values). We’ll extend this to also fill the `sleep_stages` dict if stage-wise minutes are available. For example, Apple’s JSON might include a breakdown of minutes in Core/Deep/REM – we can map those into `SleepData.sleep_stages` (e.g. `{SleepStage.LIGHT: 240, SleepStage.DEEP: 120, SleepStage.REM: 90, SleepStage.AWAKE: 30}` minutes). If Apple doesn’t directly give the breakdown, we can derive it by summing durations of each categorical segment (the HealthKit samples for each stage).
* If Apple’s “inBed” vs “asleep” distinction is important (HealthKit can log “In Bed” periods separate from “Asleep” periods), we interpret “In Bed” time as the interval from `sleep_start` to `sleep_end`, and “Asleep” as the subset of that actually asleep. Our `sleep_efficiency` can then be computed as `total_sleep_minutes / time_in_bed_minutes`. We may consider adding a field `time_in_bed_minutes` to `SleepData` for clarity – or simply compute it on the fly (since `sleep_start` and `sleep_end` are stored). The HealthKit integration already captures `HKQuantityTypeIdentifierTimeInBed` under SLEEP\_TYPES mapping, so we have that data if needed.

In summary, **no major schema changes** are needed – our existing model can represent the required sleep info. We will just make sure to **populate all relevant fields** (especially `sleep_stages` breakdown and efficiency) during ingestion or in SleepProcessor.

### **3. Pipeline Injection**

Next, integrate the SleepProcessor into the **analysis pipeline**. In `HealthAnalysisPipeline` (our orchestrator class in `analysis_pipeline.py`), we already organize incoming metrics by modality (cardio, respiratory, activity, etc.) including a `"sleep"` bucket. We need to:

* **Instantiate the SleepProcessor** in the pipeline’s `__init__`. For example:

  ```python
  from clarity.ml.processors.sleep_processor import SleepProcessor
  ...
  self.sleep_processor = SleepProcessor()
  logger.info("✅ %s initialized", self.sleep_processor.processor_name)
  ```

  similar to how `activity_processor` is added.

* **Process sleep metrics if present.** In `HealthAnalysisPipeline.process_health_data()`, after preprocessing, we have an `organized_data` dict of metrics by type. We will add a block:

  ```python
  if organized_data["sleep"]:
      self.logger.info("Processing sleep data...")
      sleep_features = self.sleep_processor.process(organized_data["sleep"])
      results.sleep_features = sleep_features
      modality_features["sleep"] = []  # (if we decide not to produce an embedding vector for sleep)
  ```

  This ensures the SleepProcessor runs and its results are stored. In this design, we might not produce a numeric embedding vector for sleep to include in `modality_features` (since there’s no pretrained transformer output for sleep stages akin to PAT). We can either leave `modality_features["sleep"]` empty or compute a simplistic vector of key sleep metrics if we want to include sleep in the fusion step. Given our current pipeline’s focus (activity/PAT is the main ML model), it may be fine to exclude sleep from the transformer fusion and just handle it via summary stats and insights. (If needed in the future, one could imagine training a small transformer to embed sleep stage sequences, but that’s beyond scope – and Apple’s data is already an analysis result, not raw signals.)

* **Incorporate Sleep in Summary Statistics:** The pipeline’s `_generate_summary_stats()` will be extended to include sleep metrics in the `health_indicators`. We can add a section like:

  ```python
  if sleep_features:
      sleep_health = {}
      for feature in sleep_features:
          name = feature["feature_name"]
          val = feature["value"]
          if name in ("average_sleep_duration", "total_sleep_time"):
              sleep_health["avg_sleep_hours"] = round(val, 2)
          elif name == "sleep_efficiency":
              sleep_health["sleep_efficiency_pct"] = int(val * 100)
          elif name == "sleep_consistency_score":
              sleep_health["consistency_score"] = round(val, 2)
          # ... and so on for latency, wake_count, etc.
      if sleep_health:
          summary["health_indicators"]["sleep_health"] = sleep_health
  ```

  This parallels how activity\_health is handled. The idea is to surface a few key sleep stats in the summary for downstream use or quick API retrieval.

* **Persisting results:** The `AnalysisResults` dataclass might get a new field (e.g. `sleep_features`). In the Firestore save step, we’ll include `results.sleep_features` in the saved dictionary (similar to `activity_features`). This way, the asynchronous insight generator (or any result fetch API) can access the sleep metrics.

With these changes, whenever HealthKit sleep data is uploaded via `/health-data`, the pipeline will output structured sleep insights alongside the existing activity, cardio, etc.

### **4. API Endpoints and Workflow**

We should consider if a new **microservice endpoint** is needed for sleep analysis or if it fits into existing ones:

* **Data Ingestion**: The existing `POST /health-data` endpoint already supports uploading sleep data. It allows multiple metrics in one upload (e.g., heart\_rate, activity\_level, sleep\_analysis, etc.). We will continue to use this endpoint for receiving HealthKit JSON. The payload is a `HealthDataUpload` with a list of metrics, each having `metric_type` and data; for sleep, `metric_type` would be `"sleep_analysis"` with `sleep_data` populated. The endpoint will invoke `HealthDataService` which likely calls our pipeline. No change needed in the API signature, but we should verify that the service triggers the pipeline for the new sleep metrics (likely it does since we handle them in the pipeline).

* **Processing and Async Handling**: If the pipeline runs asynchronously, the `/health-data` POST returns a `processing_id` and status (e.g. “processing”). Once analysis is complete, results are saved and possibly a Pub/Sub event triggers insight generation. This flow remains – we’re simply adding more content to the results.

* **Retrieval & Insights**: We have a `/pat/analysis/{id}` GET for actigraphy results already, but for sleep we might not need a separate endpoint. Instead, the combined analysis results (including sleep\_features and updated summary\_stats) can be fetched via a general endpoint if one exists (perhaps `/health-data/{id}` to get the processed result). In *Clarity Loop*, it looks like insights are primarily delivered via the Gemini service rather than raw data to the user, so we might skip adding a dedicated “/sleep-analysis” REST endpoint unless needed for debugging. If desired, a `GET /sleep/{analysis_id}` could fetch just sleep-specific metrics from the results, but this is optional.

In short, **no new public API routes are strictly required** – we integrate sleep into the existing upload/processing workflow. The key is to ensure the pipeline and subsequent insight generation include the new SleepProcessor output.

### **5. Gemini Prompt & Template Updates**

Our insight generation uses the **Gemini 2.5 LLM** (via Vertex AI) to turn analysis results into user-friendly narratives. The prompt template in `gemini_service.py` currently expects certain fields in `analysis_results` dict, including: `sleep_efficiency`, `circadian_rhythm_score`, `depression_risk_score`, `total_sleep_time`, `wake_after_sleep_onset`, and `sleep_onset_latency`. These are used to fill a structured prompt:

```text
PATIENT DATA:
- Sleep Efficiency: X%
- Circadian Rhythm Score: Y
- Depression Risk Score: Z
- Total Sleep Time: A hours
- Wake After Sleep Onset: B minutes
- Sleep Onset Latency: C minutes
...
```

with some clinical guidelines and an expected JSON format response.

To incorporate the SleepProcessor outputs (which, in our design, reflect actual HealthKit sleep data rather than PAT’s estimation), we will **populate these fields accordingly**:

* **Sleep efficiency**: If we have a reliable value from SleepData (and time in bed), we should use that to set `analysis_results["sleep_efficiency"]`. Apple doesn’t directly give efficiency, so our SleepProcessor can compute it (as noted above). We’d express it as a percentage (0–100). For example, if the user slept 7h out of 8h in bed, efficiency = 87.5%. In the prompt, this appears as “87.5%”.

* **Total Sleep Time**: Use the nightly total from SleepData (we’ll convert minutes to hours with one decimal). This populates `total_sleep_time` in the prompt. For example, 435 minutes becomes 7.3 hours.

* **WASO and Sleep Latency**: Apple’s data may not explicitly label these, but we can derive them. Our SleepProcessor can estimate *sleep onset latency* as the gap between going to bed (`sleep_start`) and first “asleep” epoch – if we have minute-level stage data, find the first non-awake minute. *Wake After Sleep Onset* can be the sum of all awake minutes between sleep onset and final awakening. We will set these in the results if possible. (If not available, we might leave the defaults or use PAT’s values as fallback. But since the user specifically wants to leverage HealthKit-like data, we’ll try to compute these from the stage timeline.)

* **Sleep stage insights**: Currently the prompt doesn’t explicitly list REM or Deep percentages. We may not add them directly to the prompt variables to keep it concise, but we **do expect Gemini to mention them** in the narrative if significant. We can encourage this by including context in the prompt’s “Additional Context” or by tweaking instructions. For example, if we detect an imbalance (say REM % is low), we could append to the `context` field of the `HealthInsightRequest` something like: *“Patient spent 15% in REM (below average) and 20% in deep sleep.”* This gives the LLM a hint to comment on it. Another approach is expanding the prompt template to list stage percentages, but that might complicate the format. A lightweight solution is fine: the Gemini prompt already includes a placeholder for `context`, so we can supply stage distribution info there.

Aside from data insertion, the **clinical guidelines** in the prompt may be updated if needed. Right now they cover sleep efficiency and circadian score ranges. We might add a guideline for sleep duration (e.g. “7-9 hours = optimal for adults”) or for wake episodes (but not strictly necessary – the LLM can handle those with general knowledge).

Finally, ensure that the **Gemini response parser** still works (it expects valid JSON as response). That remains unchanged.

### **6. End-to-End Testing Strategy**

To validate the new SleepProcessor and its integration:

* **Unit Test the Processor:** Feed a known set of sleep metrics into `SleepProcessor.process` and verify the output. For example, simulate a single-night SleepData: 8 hours in bed, 6.5 hours asleep, 3 awakenings (each \~5 min), REM 90 min, Deep 60 min, etc. Check that the processor returns correct totals (e.g. total\_sleep\_time \~6.5h, efficiency \~81%, wake\_count 3, WASO \~15 min, etc.) and that consistency score is reasonable (for one night, maybe default to 1 or None).

* **Integration Test with Sample HealthKit JSON:** Construct a mini HealthKit JSON payload similar to what the iOS app would send. For instance:

  ```json
  {
    "user_id": "...",
    "metrics": [
      {
        "metric_type": "sleep_analysis",
        "sleep_data": {
          "total_sleep_minutes": 420,
          "sleep_start": "2025-06-01T23:30:00Z",
          "sleep_end": "2025-06-02T07:30:00Z",
          "sleep_efficiency": 0.875,
          "time_to_sleep_minutes": 15,
          "wake_count": 2,
          "sleep_stages": {
            "light": 300,
            "deep": 90,
            "rem": 30,
            "awake": 0
          }
        },
        "device_id": "Apple Watch 8"
      }
    ],
    "upload_source": "apple_health",
    "client_timestamp": "2025-06-02T12:00:00Z"
  }
  ```

  Post this to the `/health-data` endpoint (in a test context). Then either poll the `/health-data/{processing_id}` or check the Firestore (if our pipeline writes to it) to get the `summary_stats`. Verify that `summary_stats["health_indicators"]["sleep_health"]` reflects the input (e.g. avg\_sleep\_hours \~7.0, sleep\_efficiency\_pct 88, etc.), and that the PAT-related fields remain consistent or are overridden appropriately.

* **Insight Generation Test:** Using the same sample, call the Gemini insight generator (directly via the `GeminiService.generate_health_insights` or via a Pub/Sub trigger if configured). Ensure the `HealthInsightRequest.analysis_results` includes our sleep metrics. Then inspect the `HealthInsightResponse`:

  * **Narrative** should mention sleep aspects – e.g. *“You slept about 7 hours with \~88% efficiency. You had 2 awakenings during the night, and spent about 15% of your sleep in deep sleep and 5% in REM, which is a bit low on REM.”* (The exact wording will vary, but it should correctly reflect the data.)
  * **Key insights** might include a bullet about consistency if multiple nights were provided (or note that sleep efficiency is good but REM is low, etc.).
  * **Recommendations** might mention sleep hygiene, since the prompt’s clinical guidelines emphasize efficiency and circadian rhythm.

* **Regression checks:** Also test that sending only activity data (no sleep) still works (to ensure our changes didn’t break the PAT pathway). And test combined uploads (sleep + activity together) to ensure both processors run and their results co-exist.

Through these tests, we can iterate on any discrepancies. For example, if Gemini misinterprets or if the JSON format is slightly off, we’ll adjust accordingly (maybe formatting the values or tweaking the prompt). Our design leverages existing structures (Pydantic models, pipeline hooks, LLM prompt variables) to minimize risk, making the SleepProcessor a **modular addition** to the system.

## Sources

* Fox *et al.* (2024), *“A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages.”* – **PFTSleep** model (8-channel, 8-hour transformer) achieves AUROC 0.95–0.99 on five-class staging, improving on prior state-of-art.

* Carter *et al.* (CVPR 2024), *“SleepVST: Sleep Staging from Near-Infrared Video Signals using Pre-Trained Transformers.”* – Transformer pre-trained on SHHS/MESA signals, adapted to 4-class (Wake/Core/Deep/REM) staging. Reports κ≈0.75 on wearables data and \~79% accuracy (κ=0.71) on 4-class video-based staging, aligning with Apple’s sleep categories.

* Phan *et al.* (2022), *“SleepTransformer: Automatic Sleep Staging with Interpretability and Uncertainty Quantification,”* IEEE TBME. – Sequence-to-sequence transformer on single-channel EEG; \~81.4% accuracy on Sleep-EDF expanded. Open-source code available, demonstrating viable performance and clinical interpretability, though on an older tech stack.

* **Clarity Backend Code References:** Our integration plan leverages the existing backend structure:

  * *HealthKit mapping and data models* – HealthKit sleep stage constants (Core/Deep/REM) and `SleepData` schema fields.
  * *ActivityProcessor example* – showing how a domain-specific processor is implemented and integrated.
  * *Analysis pipeline* – where processors are invoked and summary stats are compiled.
  * *Gemini prompt template* – current usage of sleep-related metrics in LLM prompt, which we align with our new SleepProcessor outputs.
