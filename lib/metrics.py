import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

def rmse(y, yhat): return float(np.sqrt(((y - yhat)**2).mean()))
def rmspe(y, yhat, eps=1e-8): return float(np.sqrt((((y - yhat)/(y+eps))**2).mean()))

def calibration(y, yhat):
    X = sm.add_constant(yhat.values)
    mod = sm.OLS(y.values, X).fit()
    return float(mod.params[1]), float(mod.params[0])

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h:int=1):
    d = (e1**2 - e2**2).astype(float)
    ols = sm.OLS(d, np.ones_like(d)).fit(cov_type='HAC', cov_kwds={'maxlags': max(1,h-1)})
    t = float(ols.tvalues[0])
    p = 2.0 * stats.t.sf(abs(t), df=len(d)-1)
    return float(p)