import numpy as np, pandas as pd

def block_bootstrap_indices(n, b, rng):
    starts = rng.integers(0, n - b + 1, size=(n // b + 2))
    idx = np.concatenate([np.arange(s, s+b) for s in starts])
    return idx[:n]

def estimate_bias_variance(predict_fn, Xtest, ytest, refit_fn, n_train, B=100, block_len=10, seed=1337):
    rng = np.random.default_rng(seed)
    preds = np.zeros((B, len(Xtest)))
    for i in range(B):
        idx = block_bootstrap_indices(n_train, block_len, rng)
        preds[i] = refit_fn(idx)(Xtest)
    fbar = preds.mean(0)
    bias2 = float(((fbar - ytest.values)**2).mean())
    var   = float(preds.var(0, ddof=1).mean())
    # Correct MSE: average squared error over bootstrap reps and test points
    mse   = float(((preds - ytest.values)**2).mean())
    noise = max(mse - bias2 - var, 0.0)
    return {"bias2":bias2, "var":var, "noise":noise, "mse":mse}
