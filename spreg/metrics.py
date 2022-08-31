__author__ = "Tyler D. Hoffman tdhoffman@asu.edu"

"""
Implements diagnostics not present in sklearn.metrics
"""
import numpy as np
from scipy.stats import chi2
from .utils import spdot
from libpysal.weights import lag_spatial


def chisqprob(chisq, df):
    return chi2.sf(chisq, df)


def lmtest(y_true, y_pred, X, w, tests='all'):
    if 'all' in tests:
        tests = ['lme', 'lml', 'rlme', 'rlml', 'sarma']


def _lm_err_test(y_true, y_pred, w):
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    resX = np.dot(u.T, wu) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal.sum()

    lm = resX ** 2 / trW
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def _lm_lag_test(y_true, y_pred, X, w):
    u = y_true - y_pred
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal.sum()

    n = y_pred.shape[0]
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


def _rlm_err_test(y_true, y_pred, X, w):
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    n = y_pred.shape[0]
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


def _rlm_lag_test(y_true, y_pred, X, w):
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    n = y_pred.shape[0]
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


def _lm_sarma_test(y_true, y_pred, X, w):
    u = y_true - y_pred
    wu = lag_spatial(w, u)
    resX = np.dot(u.T, wu) / sig2n
    resY = np.dot(u.T, lag_spatial(w, y_true)) / sig2n

    prod = (w.sparse.T + w.sparse) * w.sparse
    trW = prod.diagonal().sum()

    n = y_pred.shape[0]
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
