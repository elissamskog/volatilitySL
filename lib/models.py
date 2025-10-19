import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from arch.univariate import ConstantMean, GARCH, StudentsT, Normal

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
    am = ConstantMean(train_ret.dropna().astype(float))
    am.volatility = GARCH(1,0,1)
    am.distribution = StudentsT() if dist=="t" else Normal()
    res = am.fit(disp="off")
    v = (res.conditional_volatility**2).rolling(H, min_periods=H).sum()**0.5
    pred = v.iloc[len(train_ret):]  # align naïvely in WF loop
    if out_index is not None:
        pred = pred.head(len(out_index))
        pred.index = out_index[:len(pred)]
    return pred

def gbm_fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, H: int, params: dict):
    cols = [c for c in ['lag_rv_1','lag_rv_5','lag_rv_60','log_vol','trades','minute_of_day','dow'] if c in train_df]
    Xtr, ytr = train_df[cols].dropna(), train_df.loc[train_df[cols].dropna().index, f'rv_{H}']
    Xte = test_df[cols].dropna()
    if len(Xtr)==0 or len(Xte)==0:
        return pd.Series(index=test_df.index, dtype=float)

    # Merge defaults with user params (user params override defaults)
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