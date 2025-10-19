Prereqs
- Python 3.10+ in a venv with dependencies from `requirements.txt` installed.
- A checkout of this repository with data under `data/raw/` (train/test/sample_submission as provided).
- Parquet I/O requires an extra dependency such as pyarrow.

Notes
- On macOS set BLAS/OMP env vars to 1 to avoid oversubscription:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
- Scripts append to `artifacts/predictions.csv` and `reports/tables/bias_variance_summary.csv`.

	1.	Feature generation –
01_make_features.py converts raw Optiver data into per-stock feature tables stored under data/processed/.
	2.	Model training and walk-forward prediction –
02_run_models_walkforward.py fits each volatility model (Rolling Mean, HAR, GARCH, GBM) in an expanding-window scheme and produces out-of-sample forecasts saved to artifacts/predictions.csv.
	3.	Evaluation and statistical testing –
03_evaluate.py computes RMSE, RMSPE, calibration regressions, and Diebold–Mariano tests for equal predictive accuracy.
Results are written to reports/tables/.
	4.	Bias–variance decomposition –
04_bias_variance.py performs block-bootstrap resampling to estimate the bias, variance, and noise components of forecast error for each model.
The script appends results per stock and horizon to reports/tables/bias_variance_summary.csv.
	5.	Notebook summary and visualization –
The Jupyter notebook notebook.ipynb aggregates all outputs, computes summary tables by model and horizon, and generates the figures included in the report (saved under reports/figures/).
