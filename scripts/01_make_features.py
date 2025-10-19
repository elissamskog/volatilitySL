import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "lib"))

import argparse, os, pandas as pd
from pathlib import Path
from io_utils import load_settings, ensure_dirs, read_any
from features import make_minute_bars, build_features

parser = argparse.ArgumentParser()
parser.add_argument("--settings", default="settings.yaml")
parser.add_argument("--stock-limit", type=int, default=None) # limit for quick testing
args = parser.parse_args()

cfg = load_settings(args.settings)
paths = cfg["paths"]; ensure_dirs(paths["processed"])

trade_path = Path(paths["raw"]) / "trade_train.parquet"
trade = read_any(trade_path)

if "stock_id" in trade.columns:
    stocks = trade["stock_id"].unique().tolist()
    split_col = "stock_id"
elif "__source" in trade.columns:
    stocks = trade["__source"].unique().tolist()
    split_col = "__source"
else:
    stocks = [0]
    split_col = None

all_out = []

if args.stock_limit is not None:
    stocks = stocks[: args.stock_limit]

for sid in stocks:
    if split_col is None:
        dft = trade.copy()
    else:
        dft = trade[trade[split_col] == sid]
    bars = make_minute_bars(dft)
    feats = build_features(bars)
    out = Path(paths["processed"]) / f"features_stock_{sid}.csv"

    if out.exists():
        print("skipping (exists)", out)
        all_out.append(str(out))
        continue
    feats.to_csv(out, index=False)
    all_out.append(str(out))
    print("saved", out)

print("done. files:", len(all_out))