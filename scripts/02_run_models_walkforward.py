import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "lib"))

import argparse, pandas as pd
from pathlib import Path
from io_utils import load_settings, ensure_dirs
from models import rolling_mean, har_fit_predict, garch_fit_predict, gbm_fit_predict

parser = argparse.ArgumentParser()
parser.add_argument("--settings", default="settings.yaml")
parser.add_argument("--stock-limit", type=int, default=None, help="Limit number of stocks to process (dev)")
args = parser.parse_args()

# Use to avoid zsh kill due to excessive threads on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

cfg = load_settings(args.settings); paths = cfg["paths"]
ensure_dirs(paths["artifacts"])

feat_files = sorted(Path(paths["processed"]).glob("features_stock_*.csv"))
if args.stock_limit is not None:
    feat_files = feat_files[: args.stock_limit]

out = Path(paths["artifacts"]) / "predictions.csv"
write_header = not out.exists()

for fp in feat_files:
    stock = fp.stem.split("_")[-1]
    df = pd.read_csv(fp, parse_dates=["minute"])
    rows = []
    for H in cfg["horizons"]:
        target = f"rv_{H}"
        step, window, val_tail = cfg["splits"].values()
        for start in range(0, max(0, len(df)-window-step), step):
            train = df.iloc[start:start+window].copy()
            test  = df.iloc[start+window:start+window+step].copy()

            # rolling baseline hyperparam via last val_tail
            best_L, best_rmse = None, 1e9
            for L in cfg["models"]["rolling"]["L_grid"]:
                val = train.iloc[-val_tail:]
                pred = rolling_mean(val[target], L)
                m = (val[target] - pred).dropna()
                if len(m)==0: continue
                rmse = (m.pow(2).mean())**0.5
                if rmse < best_rmse: best_rmse, best_L = rmse, L
            pred_roll = rolling_mean(pd.concat([train, test])[target], best_L).iloc[len(train):]
            pred_roll.index = test.index[:len(pred_roll)]

            pred_har  = har_fit_predict(train, pd.concat([train.tail(200), test]), H).loc[test.index.intersection(train.index.union(test.index))]
            pred_garch= garch_fit_predict(train['ret'], H, cfg["models"]["garch"]["dist"], out_index=test.index)
            pred_gbm  = gbm_fit_predict(train, test, H, cfg["models"]["gbm"])

            y_true = test[target]
            for name, pred in [("rolling", pred_roll), ("har", pred_har), ("garch", pred_garch), ("gbm", pred_gbm)]:
                aligned = pd.concat([y_true, pred], axis=1, keys=["y","yhat"]).dropna()
                for idx, r in aligned.iterrows():
                    rows.append({"stock":stock,"horizon":H,"minute":idx,"model":name,"y":float(r.y),"yhat":float(r.yhat)})

    if len(rows) > 0:
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, mode="a", header=write_header, index=False)
        write_header = False
        print(f"appended {len(df_out)} rows for {stock} -> {out}")

print("done, saved", out)