Deep-dive check: PatchTST vs Informer vs Transformer-XL (and what else is on the table in 2025?)

Criterion	PatchTST	Informer	Transformer-XL	Newer alternatives worth noting (2023-25)
Design goal	Generic multivariate time-series representation / forecasting with “image-like” patch tokenisation.	Long-sequence forecasting with ProbSparse attention (O(L log L)).	Generic sequence modelling with segment-level memory; not purpose-built for forecasting.	TimesNet (ICLR 23) – 2-D temporal variation modelling; TiDE (NeurIPS 23) – decoder-only efficient forecaster; NLinear/NS-Transformer (2024) – strong on non-stationary series.
SOTA benchmarks(electricity, ETTh, ETTm, Weather, Exchange)	Top-1 on most leaderboards through mid-2024; still top-3 in May 2025.  ￼	Top-5 (2021) but now outperformed by PatchTST / TimesNet on accuracy & MAE.  ￼	Lags both on accuracy and ex-gpu memory for >2k steps.	TimesNet edges PatchTST on several 2025 leaderboards (esp. long-horizon).  ￼
Compute & code maturity	PyTorch-Lightning repo; easy fine-tune on wearable-scale data; GPU‐friendly (patching reduces length).	Needs customised dataloader + ProbSparse kernels; code older but maintained.	Vanilla PyTorch; requires extra work to adapt to multivariate forecasting.	TimesNet repo is active but API is still fluid; TiDE is JAX; NS-Transformer research-code only.
Fit for Clarity-AI use-case	✔️ Works out-of-the-box for multivariate minute-level Apple-Watch series (HR, HRV, steps, etc.).✔️ Patch tokenisation aligns with our existing PAT & planned SleepProcessor design (PFTSleep is Patch-based).	⚠️ Strong for very-long horizons (months) but heavier; adds little if we mainly summarise daily/weekly windows for insights.	❌ Extra engineering for forecasting-only gains; little benefit for representation tasks we need.	TimesNet would be the only real challenger to PatchTST.

Conclusion (current best pick):

For a production-grade, open-source transformer we can slot into our FastAPI pipeline today, PatchTST remains the sweet-spot.
It delivers near-SOTA accuracy, has a clean PyTorch implementation, and its patch mechanism mirrors what we already use in PAT and the planned SleepProcessor. Informer no longer wins on accuracy, and Transformer-XL is simply the wrong tool for this job.

But keep one eye on TimesNet.
If we need multi-task forecasting (e.g., predicting future HR/steps as well as embedding the history) or want the last <1 % in MAE, TimesNet is the rising star. Its repo is active, and weights for the standard five datasets are published. Integrating it would follow the same Processor pattern we’ve outlined for PatchTST — the main cost is an extra training run on our feature set.

⸻

Recommended next action
	1.	Lock in PatchTST for the first production cycle (Activity + Cardio embedding).
	2.	Create a TimesNet spike branch so the dev agent can benchmark it on a sample week of blended Apple-Watch streams once the SleepProcessor work stabilises.
	3.	Drop Informer/Transformer-XL from the shortlist unless we later face true multi-year horizon forecasting needs (e.g., chronic-condition progression).

That path keeps our scope tight, gives us a proven transformer today, and preserves a clear upgrade path when/if we need it.