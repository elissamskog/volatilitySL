import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from arch.univariate import ConstantMean, GARCH, StudentsT, Normal
import warnings
from arch.univariate import base as arch_base

def rolling_mean(y: pd.Series, L: int) -> pd.Series:
    return y.rolling(L, min_periods=L).mean()

def har_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, H: int, l2=0.0):
    cols = [c for c in ['lag_rv_1','lag_rv_5','lag_rv_60'] if c in train_df]
    Xtr, ytr = train_df[cols].dropna(), train_df.loc[train_df[cols].dropna().index, f'rv_{H}']
    Xte = test_df[cols].dropna()
    model = Ridge(alpha=l2)
    model.fit(Xtr, ytr)
    return pd.Series(model.predict(Xte), index=Xte.index)

def garch_fit_predict(train_ret: pd.Series, H: int, dist="normal", out_index=None):
    tr = train_ret.dropna().astype(float)
    if len(tr) == 0:
        return pd.Series(index=out_index if out_index is not None else [])

    # Rescale series to avoid warnings
    scale = 1.0
    cur_scale = float((tr.abs().mean()))
    if cur_scale > 0 and not (1.0 <= cur_scale <= 1000.0):
        target = 100.0
        scale = target / cur_scale
        tr_scaled = tr * scale
    else:
        tr_scaled = tr.copy()

    am = ConstantMean(tr_scaled)
    am.volatility = GARCH(1, 0, 1)
    am.distribution = StudentsT() if dist == "t" else Normal()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=arch_base.DataScaleWarning)
        res = am.fit(disp="off")

    try:
        fc = res.forecast(horizon=H, reindex=False)
        var_col = f"h.{H}"
        if hasattr(fc, "variance") and var_col in fc.variance.columns:
            var_fc = float(fc.variance.iloc[-1][var_col])
            if cur_scale > 0 and not (1.0 <= cur_scale <= 1000.0):
                var_fc = var_fc / (scale ** 2)
            vol_fc = var_fc ** 0.5
            if out_index is not None:
                return pd.Series([vol_fc] * len(out_index), index=out_index)
            else:
                return pd.Series([vol_fc])
    except Exception:
        pass

    v = (res.conditional_volatility ** 2).rolling(H, min_periods=H).sum() ** 0.5
    pred = v.iloc[len(tr):]
    if out_index is not None:
        pred = pred.head(len(out_index))
        pred.index = out_index[: len(pred)]
    return pred

def gbm_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, H: int, params: dict):
    cols = [c for c in ['lag_rv_1','lag_rv_5','lag_rv_60','log_vol','trades','minute_of_day','dow'] if c in train_df]
    Xtr, ytr = train_df[cols].dropna(), train_df.loc[train_df[cols].dropna().index, f'rv_{H}']
    Xte = test_df[cols].dropna()
    if len(Xtr)==0 or len(Xte)==0:
        return pd.Series(index=test_df.index, dtype=float)

    params = params or {}
    kwargs = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=0,
    )
    kwargs.update(params)

    model = XGBRegressor(**kwargs)
    model.fit(Xtr, ytr)
    return pd.Series(model.predict(Xte), index=Xte.index)