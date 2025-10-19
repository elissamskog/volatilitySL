import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "lib"))

import pandas as pd
from pathlib import Path
from io_utils import load_settings, ensure_dirs
from metrics import rmse, rmspe, calibration, diebold_mariano

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="settings.yaml")
    args = parser.parse_args()

    cfg = load_settings(args.settings)
    paths = cfg["paths"]
    ensure_dirs(paths["tables"])

    pred_path = Path(paths["artifacts"]) / "predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"Missing {pred_path}. Run scripts/02_run_models_walkforward.py first.")

    df = pd.read_csv(pred_path)
    if "minute" in df.columns:
        try:
            df["minute"] = pd.to_datetime(df["minute"])
        except Exception:
            pass

    # Metrics per horizon and model
    rows = []
    for (h, m), g in df.groupby(["horizon", "model"]):
        y, yhat = g["y"], g["yhat"]
        R = {
            "horizon": h,
            "model": m,
            "RMSE": rmse(y, yhat),
            "RMSPE": rmspe(y, yhat),
        }
        try:
            slope, intercept = calibration(y, yhat)
        except Exception:
            slope, intercept = float("nan"), float("nan")
        R["CalSlope"] = slope
        R["CalIntercept"] = intercept
        rows.append(R)

    metrics_df = pd.DataFrame(rows).sort_values(["horizon", "RMSE"])
    out_metrics = Path(paths["tables"]) / "metrics_summary.csv"
    metrics_df.to_csv(out_metrics, index=False)
    print("Saved:", out_metrics)

    # DM pairwise tests per horizon
    models = df["model"].unique().tolist()
    dm_rows = []
    for h, gh in df.groupby("horizon"):
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                g1 = gh[gh["model"] == m1].copy()
                g2 = gh[gh["model"] == m2].copy()
                merged = g1.merge(
                    g2,
                    on=["stock", "minute", "horizon"],
                    suffixes=("_1", "_2"),
                    how="inner",
                )
                if len(merged) < 10:
                    continue
                e1 = (merged["y_1"] - merged["yhat_1"]).values
                e2 = (merged["y_2"] - merged["yhat_2"]).values
                try:
                    p = diebold_mariano(e1, e2, h=1)
                except Exception:
                    p = float("nan")
                dm_rows.append({"horizon": h, "m1": m1, "m2": m2, "p_value": p})

    dm_df = pd.DataFrame(dm_rows)
    out_dm = Path(paths["tables"]) / "dm_tests.csv"

    dm_df.to_csv(out_dm, index=False, float_format="%.12e")
    print("Saved:", out_dm)

if __name__ == "__main__":
    main()