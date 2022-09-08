__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

"""
Implements diagnostics not present in sklearn.metrics
"""
import numpy as np
from scipy.stats import chi2
from ..utils import spdot
from libpysal.weights import lag_spatial

ALL_TESTS = ['lme', 'lml', 'rlme', 'rlml', 'sarma']


def chisqprob(chisq, df):
    return chi2.sf(chisq, df)


def lm_test(y_true, y_pred, X, w, tests=['all']):
    if 'all' in tests:
        tests = ['lme', 'lml', 'rlme', 'rlml', 'sarma']

    out = dict.fromkeys(tests)
    for test in tests:
        if test == "lme":
            out[test] = lm_err_test(y_true, y_pred, w)
        elif test == "lml":
            out[test] = lm_lag_test(y_true, y_pred, X, w)
        elif test == "rlme":
            out[test] = rlm_err_test(y_true, y_pred, X, w)
        elif test == "rlml":
            out[test] = rlm_lag_test(y_true, y_pred, X, w)
        elif test == "sarma":
            out[test] = lm_sarma_test(y_true, y_pred, X, w)
        else:
            raise ValueError(f"Test must be one of {ALL_TESTS}, was {test}")

    return out


def lm_err_test(y_true, y_pred, w):
    n = y_true.shape[0]
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    sig2n = np.dot(u.T, u) / n
    resX = np.dot(u.T, wu) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    lm = resX ** 2 / trW
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lm_lag_test(y_true, y_pred, X, w):
    n = y_true.shape[0]
    u = y_true - y_pred
    sig2n = np.dot(u.T, u) / n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    wxb = lag_spatial(w, y_pred)
    wxb2 = np.dot(wxb.T, wxb)
    xwxb = spdot(X.T, wxb)
    num1 = wxb2 - np.dot(xwxb.T, np.dot(np.linalg.inv(np.dot(X.T, X)), xwxb))
    num = num1 + (trW * sig2n)
    den = n / sig2n
    j = num / den

    lm = resY ** 2 / (n * j)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlm_err_test(y_true, y_pred, X, w):
    n = y_true.shape[0]
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    sig2n = np.dot(u.T, u) / n
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    wxb = lag_spatial(w, y_pred)
    wxb2 = np.dot(wxb.T, wxb)
    xwxb = spdot(X.T, wxb)
    num1 = wxb2 - np.dot(xwxb.T, np.dot(np.linalg.inv(np.dot(X.T, X)), xwxb))
    num = num1 + (trW * sig2n)
    den = n / sig2n
    j = num / den

    nj = n * j
    num = (resX - (trW * resY) / nj) ** 2
    den = trW * (1. - (trW / nj))
    lm = num / den
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlm_lag_test(y_true, y_pred, X, w):
    n = y_true.shape[0]
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    sig2n = np.dot(u.T, u) / n
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    wxb = lag_spatial(w, y_pred)
    wxb2 = np.dot(wxb.T, wxb)
    xwxb = spdot(X.T, wxb)
    num1 = wxb2 - np.dot(xwxb.T, np.dot(np.linalg.inv(np.dot(X.T, X)), xwxb))
    num = num1 + (trW * sig2n)
    den = n / sig2n
    j = num / den

    lm = (resY - resX) ** 2 / \
        ((n * j) - trW)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lm_sarma_test(y_true, y_pred, X, w):
    n = y_true.shape[0]
    u = y_true - y_pred
    sig2n = np.dot(u.T, u) / n
    wu = lag_spatial(w, u)
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    wxb = lag_spatial(w, y_pred)
    wxb2 = np.dot(wxb.T, wxb)
    xwxb = spdot(X.T, wxb)
    num1 = wxb2 - np.dot(xwxb.T, np.dot(np.linalg.inv(np.dot(X.T, X)), xwxb))
    num = num1 + (trW * sig2n)
    den = n / sig2n
    j = num / den

    first = (resY - resX) ** 2 / \
        (n * j - trW)
    secnd = resX ** 2 / trW
    lm = first + secnd
    pval = chisqprob(lm, 2)
    return (lm[0][0], pval[0][0])


if __name__ == "__main__":
    import spreg
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from libpysal.examples import load_example
    from libpysal.weights import Kernel, fill_diagonal

    boston = load_example("Bostonhsg")
    boston_df = gpd.read_file(boston.get_path("boston.shp"))

    boston_df["NOXSQ"] = (10 * boston_df["NOX"])**2
    boston_df["RMSQ"] = boston_df["RM"]**2
    boston_df["LOGDIS"] = np.log(boston_df["DIS"].values)
    boston_df["LOGRAD"] = np.log(boston_df["RAD"].values)
    boston_df["TRANSB"] = boston_df["B"].values / 1000
    boston_df["LOGSTAT"] = np.log(boston_df["LSTAT"].values)

    fields = ["RMSQ", "CRIM"]
    X = boston_df[fields].values
    y = np.log(boston_df["CMEDV"].values)  # predict log corrected median house prices from covars

    weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    weights = fill_diagonal(weights, 0)

    model = spreg.sklearn.Error(w=weights)
    model.fit(X, y)
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)
    print(model.score(X, y))

    old_model = spreg.GM_Error(y, X, weights)
    print(old_model.betas)

    y_pred = model.predict(X)
    print(spreg.sklearn.lm_err_test(y, y_pred, weights))
    print(spreg.sklearn.lm_lag_test(y, y_pred, X, weights))
    print(spreg.sklearn.rlm_err_test(y, y_pred, X, weights))
    print(spreg.sklearn.rlm_lag_test(y, y_pred, X, weights))
    print(spreg.sklearn.lm_sarma_test(y, y_pred, X, weights))
