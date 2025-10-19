import os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "lib"))

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from io_utils import load_settings, ensure_dirs

def bar_plot(df, x, y, title, out_path):
    plt.figure()
    df.plot(kind="bar", x=x, y=y, legend=False)
    plt.title(f"{title}: {y}")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.with_name(out_path.stem + f"_{y}.png"), dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="settings.yaml")
    args = parser.parse_args()

    cfg = load_settings(args.settings)
    paths = cfg["paths"]
    ensure_dirs(paths["figures"])

    tables = Path(paths["tables"])
    figs   = Path(paths["figures"])

    bv_path = tables / "bias_variance_summary.csv"
    if bv_path.exists():
        bv = pd.read_csv(bv_path)
        for col in ["bias2", "var", "noise", "mse"]:
            bar_plot(bv, x="horizon", y=col, title="Bias-Variance-Noise by Horizon", out_path=figs / "bv_horizon.png")
        print("Saved bias-variance figures to:", figs)
    else:
        print(f"Skipping BV plots (missing {bv_path}).")

    metrics_path = tables / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        best = metrics.sort_values(["horizon", "RMSE"]).groupby("horizon").first().reset_index()
        for col in ["RMSE", "RMSPE"]:
            bar_plot(best, x="horizon", y=col, title="Best Accuracy by Horizon", out_path=figs / "acc_horizon.png")
        print("Saved accuracy figures to:", figs)
    else:
        print(f"Skipping accuracy plots (missing {metrics_path}).")

if __name__ == "__main__":
    main()