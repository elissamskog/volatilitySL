import os, sys, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "lib"))

from pathlib import Path
from io_utils import load_settings, ensure_dirs
from bias_variance import estimate_bias_variance  # uses indices function we pass in
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def har_design(df: pd.DataFrame, H: int):
    cols = [c for c in ["lag_rv_1", "lag_rv_5", "lag_rv_60"] if c in df.columns]
    X = df[cols].copy()
    y = df[f"rv_{H}"].copy()
    mask = X.notnull().all(1) & y.notnull()
    return X[mask], y[mask]

def gbm_design(df: pd.DataFrame, H: int):
    cols = [c for c in ["lag_rv_1","lag_rv_5","lag_rv_60","log_vol","trades","minute_of_day","dow"] if c in df.columns]
    X = df[cols].copy()
    y = df[f"rv_{H}"].copy()
    mask = X.notnull().all(1) & y.notnull()
    return X[mask], y[mask]


def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Non-overlapping block bootstrap with random starts; simple but can be too stable if blocks are long."""
    starts = rng.integers(0, max(1, n - block_len + 1), size=(n // block_len + 2))
    idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])
    return idx[:n]

def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Politis–Romano stationary bootstrap: random restarts with prob p=1/mean_block and wrap-around.
    Produces more stochastic resamples for dependent data.
    """
    p = 1.0 / max(1, mean_block)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="settings.yaml")
    parser.add_argument("--feat-file", default=None, help="Path to single features CSV to analyze (overrides automatic discovery)")
    parser.add_argument("--all-stocks", action="store_true", help="Run bias-variance for all feature files under processed/ and append results")
    parser.add_argument("--test_len", type=int, default=400, help="Size of test slice from the end")
    parser.add_argument("--model", choices=["har","gbm","garch"], default="har", help="Which model to run BV on")
    parser.add_argument("--bootstrap", choices=["stationary","block"], default="stationary", help="Bootstrap type")
    parser.add_argument("--mean_block", type=int, default=5, help="Mean block length for stationary bootstrap")
    parser.add_argument("--block_len", type=int, default=None, help="Block length for simple block bootstrap (overrides settings)")
    parser.add_argument("--B", type=int, default=None, help="Number of bootstrap refits (overrides settings)")
    args = parser.parse_args()

    cfg = load_settings(args.settings)
    paths = cfg["paths"]
    ensure_dirs(paths["tables"])

    feat_files = sorted(Path(paths["processed"]).glob("features_stock_*.csv"))
    if args.feat_file:
        feat_files = [Path(args.feat_file)]
    if not feat_files:
        raise SystemExit("No features found. Run scripts/01_make_features.py first.")

    B = int(args.B if args.B is not None else cfg["block_bootstrap"]["B"])
    block_len = int(args.block_len if args.block_len is not None else cfg["block_bootstrap"]["block_len"])
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    out = Path(paths["tables"]) / "bias_variance_summary.csv"
    header_needed = not out.exists()

    for feat_fp in feat_files:
        feats = pd.read_csv(feat_fp, parse_dates=["minute"])
        stock_id = feat_fp.stem.split("_")[-1]

        rows = []
        for H in cfg["horizons"]:
            if args.model == "har":
                X, y = har_design(feats, H)
            elif args.model == "gbm":
                X, y = gbm_design(feats, H)
            else: # garch
                y = feats[f"rv_{H}"].copy()
                ret = feats["ret"].copy()
                mask = y.notnull() & ret.notnull()
                y = y[mask]
                X = pd.DataFrame(index=y.index)

            if len(X) < args.test_len + 20:
                print(f"Skipping H={H}: not enough rows for test_len={args.test_len}.")
                continue

            Xtrain, ytrain = X.iloc[:-args.test_len], y.iloc[:-args.test_len]
            Xtest, ytest   = X.iloc[-args.test_len:], y.iloc[-args.test_len:]

            if args.model == "har":
                l2 = float(cfg["models"]["har"].get("l2", 0.0))
                base = Ridge(alpha=l2, random_state=0).fit(Xtrain, ytrain)

                def predict_fn(Xtest_df):
                    return base.predict(Xtest_df)

                def refit_fn(idx_array):
                    def inner(Xtest_df):
                        m = Ridge(alpha=l2, random_state=0).fit(
                            Xtrain.iloc[idx_array], ytrain.iloc[idx_array]
                        )
                        return m.predict(Xtest_df)
                    return inner

            elif args.model == "gbm":
                gbm_params = {
                    "n_estimators": 400,
                    "max_depth": 3,
                    "learning_rate": 0.07,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "tree_method": "hist",
                    "n_jobs": 4,
                }
                base = XGBRegressor(**gbm_params).fit(Xtrain, ytrain)

                def predict_fn(Xtest_df):
                    return base.predict(Xtest_df)

                def refit_fn(idx_array):
                    def inner(Xtest_df):
                        m = XGBRegressor(**gbm_params).fit(
                            Xtrain.iloc[idx_array], ytrain.iloc[idx_array]
                        )
                        return m.predict(Xtest_df)
                    return inner

            else:  # garch
                from models import garch_fit_predict

                full_ret = feats['ret'].copy()
                ret_aligned = full_ret.reindex(y.index).dropna()

                train_ret = ret_aligned.iloc[: len(ytrain)]
                test_index = y.iloc[-args.test_len:].index

                dist = cfg.get('models', {}).get('garch', {}).get('dist', 'normal')
                base_pred = garch_fit_predict(train_ret, H, dist=dist, out_index=test_index)

                def predict_fn(Xtest_df):
                    return base_pred.reindex(Xtest_df.index).ffill().bfill().values

                def refit_fn(idx_array):
                    def inner(Xtest_df):
                        sub_train = train_ret.iloc[idx_array]
                        return garch_fit_predict(sub_train, H, dist=dist, out_index=Xtest_df.index).values
                    return inner

            if args.bootstrap == "stationary":
                indexer = lambda n, rng_local: stationary_bootstrap_indices(n, args.mean_block, rng_local)
            else:
                indexer = lambda n, rng_local: block_bootstrap_indices(n, block_len, rng_local)

            def estimate_bv_with(indexer_fn):
                preds = np.zeros((B, len(Xtest)))
                for b in range(B):
                    idx = indexer_fn(len(Xtrain), rng)
                    preds[b] = refit_fn(idx)(Xtest)
                fbar = preds.mean(0)
                bias2 = float(((fbar - ytest.values)**2).mean())
                var   = float(preds.var(0, ddof=1).mean())
                mse   = float(((preds - ytest.values[None, :])**2).mean())
                noise = max(mse - bias2 - var, 0.0)
                share = {
                    "bias_share": (bias2 / mse) if mse > 0 else np.nan,
                    "var_share":  (var   / mse) if mse > 0 else np.nan,
                    "noise_share":(noise / mse) if mse > 0 else np.nan,
                }
                return {"bias2": bias2, "var": var, "noise": noise, "mse": mse, **share}

            bv = estimate_bv_with(indexer)
            rows.append({"stock": stock_id, "horizon": H, "model": args.model, "bootstrap": args.bootstrap, **bv})

        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, mode="a", header=header_needed, index=False)
        header_needed = False
        print(f"Appended BV results for stock={stock_id} -> {out}")

    print("Saved:", out)

if __name__ == "__main__":
    main()