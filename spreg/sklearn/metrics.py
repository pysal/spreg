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


def lm_test(y_true, y_pred, w, X=None, tests=['all']):
    """
    Lagrange Multiplier tests. Implemented as presented in :cite:`Anselin1996a`.

    Parameters
    ----------
    y_true          : array
                      nx1 array of true y values
    y_pred          : array
                      nx1 array of predicted y values (from a regression)
    w               : libpysal.weights.W
                      spatial weights object
    X               : array
                      nxk array of covariates used in the model (used for all tests except
                      error; default None)
    tests           : list of strings
                      names of tests to be done (default ["all"]). available tests are:
                      "lme" (LM error), "lml" (LM lag), "rlme" (robust LM error),
                      "rlml" (robust LM lag), and "sarma" (LM SARMA). use "all"
                      to run all available tests.

    Returns
    -------
    out             : dictionary
                      contains key-value pairs corresponding to the name of the test (key)
                      and the statistic and p-value from the test (value)

    Examples
    -------

    >>> import numpy as np
    >>> import spreg.sklearn
    >>> from sklearn.linear_model import LinearRegression
    >>> import geopandas as gpd
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Kernel, fill_diagonal

    Load data.

    >>> boston = load_example("Bostonhsg")
    >>> boston_df = gpd.read_file(boston.get_path("boston.shp"))

    Transform variables prior to fitting regression.

    >>> boston_df["RMSQ"] = boston_df["RM"]**2
    >>> boston_df["LCMEDV"] = np.log(boston_df["CMEDV"])

    Set up model matrices. We're going to predict log corrected median
    house prices from the covariates.

    >>> fields = ["RMSQ", "CRIM"]
    >>> X = boston_df[fields].values
    >>> y = boston_df["LCMEDV"].values

    Create weights matrix.

    >>> weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    >>> weights = fill_diagonal(weights, 0)

    Fit OLS model.

    >>> model = LinearRegression()
    >>> model = model.fit(X, y)
    >>> print(model.intercept_)
    2.1028110827643336
    >>> print(model.coef_)
    [ 0.02508573 -0.01976315]
    >>> print(model.score(X, y))
    0.5792144821996128

    Predict new y values and look at the LM test results.

    >>> y_pred = model.predict(X)
    >>> print(spreg.sklearn.lm_test(y, y_pred, weights, X=X))
    {'lme': (798.0561649785933, 1.4278628273902582e-175), 'lml': (20.247083115373105, 6.805714788173705e-06), 'rlme': (-6395.976402372408, 1.0), 'rlml': (-7173.785484235628, 1.0), 'sarma': (-6375.729319257035, 1.0)}
    """

    if type(tests) == str:
        tests = [tests]

    if 'all' in tests:
        tests = ['lme', 'lml', 'rlme', 'rlml', 'sarma']

    if X is None:
        print("X not provided, performing only LM Error test")
        tests = ["lme"]

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
    return (lm, pval)


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
    return (lm, pval)


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
    return (lm, pval)


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
    return (lm, pval)


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
    return (lm, pval)

