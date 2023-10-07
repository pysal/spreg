"""
Diagnostics for panel data estimation
"""

__author__ = "Wei Kang weikang9009@gmail.com, \
              Pedro Amaral pedroamaral@cedeplar.ufmg.br, \
              Pablo Estrada pabloestradace@gmail.com"

import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from . import user_output as USER
from .ols import OLS
from .utils import spdot
from scipy import stats
from .panel_utils import check_panel

chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

__all__ = [
    "panel_LMlag",
    "panel_LMerror",
    "panel_rLMlag",
    "panel_rLMerror",
    "panel_Hausman",
]


def panel_LMlag(y, x, w):
    """
    Lagrange Multiplier test on lag spatial autocorrelation in panel data.
    :cite:`Anselin2008`.

    Parameters
    ----------
    y            : array
                   nxt or (nxt)x1 array for dependent variable
    x            : array
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
    w            : pysal W object
                   Spatial weights object

    Returns
    -------
    lme          : tuple
                   Pair of statistic and p-value for the LM lag test.
    """
    y, x, name_y, name_x, warn = check_panel(y, x, w, None, None)
    x, name_x, warn = USER.check_constant(x, name_x)
    ols = OLS(y, x)
    n = w.n
    t = y.shape[0] // n
    W = w.full()[0]
    Wsp_nt = sp.kron(sp.identity(t), w.sparse, format="csr")
    wxb = spdot(Wsp_nt, ols.predy)
    ww = spdot(W, W)
    wTw = spdot(W.T, W)
    trw = ww.diagonal().sum() + wTw.diagonal().sum()
    num1 = np.asarray(sp.identity(t * n) - spdot(x, spdot(ols.xtxi, x.T)))
    num2 = spdot(wxb.T, spdot(num1, wxb))
    num = num2 + (trw * trw * ols.sig2)
    J = num / ols.sig2
    utwy = spdot(ols.u.T, spdot(Wsp_nt, y))
    lm = utwy ** 2 / (ols.sig2 ** 2 * J)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def panel_LMerror(y, x, w):
    """
    Lagrange Multiplier test on error spatial autocorrelation in panel data.
    :cite:`Anselin2008`.

    Parameters
    ----------
    y            : array
                   nxt or (nxt)x1 array for dependent variable
    x            : array
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
    w            : pysal W object
                   Spatial weights object

    Returns
    -------
    lme          : tuple
                   Pair of statistic and p-value for the LM error test.
    """
    y, x, name_y, name_x, warn = check_panel(y, x, w, None, None)
    x, name_x, warn = USER.check_constant(x, name_x)
    ols = OLS(y, x)
    n = w.n
    t = y.shape[0] // n
    W = w.full()[0]
    Wsp_nt = sp.kron(sp.identity(t), w.sparse, format="csr")
    ww = spdot(W, W)
    wTw = spdot(W.T, W)
    trw = ww.diagonal().sum() + wTw.diagonal().sum()
    utwu = spdot(ols.u.T, spdot(Wsp_nt, ols.u))
    lm = utwu ** 2 / (ols.sig2 ** 2 * t * trw)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def panel_rLMlag(y, x, w):
    """
    Robust Lagrange Multiplier test on lag spatial autocorrelation in
    panel data. :cite:`Elhorst2014`.

    Parameters
    ----------
    y            : array
                   nxt or (nxt)x1 array for dependent variable
    x            : array
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
    w            : pysal W object
                   Spatial weights object

    Returns
    -------
    lme          : tuple
                   Pair of statistic and p-value for the Robust LM lag test.
    """
    y, x, name_y, name_x, warn = check_panel(y, x, w, None, None)
    x, name_x, warn = USER.check_constant(x, name_x)
    ols = OLS(y, x)
    n = w.n
    t = y.shape[0] // n
    W = w.full()[0]
    Wsp_nt = sp.kron(sp.identity(t), w.sparse, format="csr")
    wxb = spdot(Wsp_nt, ols.predy)
    ww = spdot(W, W)
    wTw = spdot(W.T, W)
    trw = ww.diagonal().sum() + wTw.diagonal().sum()
    utwu = spdot(ols.u.T, spdot(Wsp_nt, ols.u))
    num1 = np.asarray(sp.identity(t * n) - spdot(x, spdot(ols.xtxi, x.T)))
    num2 = spdot(wxb.T, spdot(num1, wxb))
    num = num2 + (t * trw * ols.sig2)
    J = num / ols.sig2
    utwy = spdot(ols.u.T, spdot(Wsp_nt, y))
    lm = (utwy / ols.sig2 - utwu / ols.sig2) ** 2 / (J - t * trw)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def panel_rLMerror(y, x, w):
    """
    Robust Lagrange Multiplier test on error spatial autocorrelation in
    panel data. :cite:`Elhorst2014`.

    Parameters
    ----------
    y            : array
                   nxt or (nxt)x1 array for dependent variable
    x            : array
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
    w            : pysal W object
                   Spatial weights object

    Returns
    -------
    lme          : tuple
                   Pair of statistic and p-value for the Robust LM error test.
    """
    y, x, name_y, name_x, warn = check_panel(y, x, w, None, None)
    x, name_x, warn = USER.check_constant(x, name_x)
    ols = OLS(y, x)
    n = w.n
    t = y.shape[0] // n
    W = w.full()[0]
    Wsp_nt = sp.kron(sp.identity(t), w.sparse, format="csr")
    wxb = spdot(Wsp_nt, ols.predy)
    ww = spdot(W, W)
    wTw = spdot(W.T, W)
    trw = ww.diagonal().sum() + wTw.diagonal().sum()
    utwu = spdot(ols.u.T, spdot(Wsp_nt, ols.u))
    num1 = np.asarray(sp.identity(t * n) - spdot(x, spdot(ols.xtxi, x.T)))
    num2 = spdot(wxb.T, spdot(num1, wxb))
    num = num2 + (t * trw * ols.sig2)
    J = num / ols.sig2
    utwy = spdot(ols.u.T, spdot(Wsp_nt, y))
    lm = (utwu / ols.sig2 - t * trw / J * utwy / ols.sig2) ** 2 / (
        t * trw * (1 - t * trw / J)
    )
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def panel_Hausman(panel_fe, panel_re, sp_lag=True):
    """
    Hausman test on panel data with spatial interactions. :cite:`Elhorst2014`.

    Parameters
    ----------
    panel_fe     : panel_fe
                   Instance from a fixed effects panel spatial regression
    panel_re     : panel_re
                   Instance from a random effects panel spatial regression
    sp_lag       : boolean
                   if True, calculate Hausman test for spatial lag model.

    Returns
    -------
    h            : tuple
                   Pair of statistic and p-value for the Hausman test.
    """
    if hasattr(panel_fe, "rho") & hasattr(panel_re, "rho"):
        d = panel_fe.betas - panel_re.betas[1:-1]
    elif hasattr(panel_fe, "lam") & hasattr(panel_re, "lam"):
        d = panel_fe.betas[0:-1] - panel_re.betas[1:-2]
    else:
        raise Exception("Only same spatial interaction allowed")

    vard = panel_re.varb[1:, 1:] - panel_fe.varb
    vardi = la.inv(vard)
    h = spdot(d.T, spdot(vardi, d))
    pval = chisqprob(h, panel_re.k)
    return (h[0][0], pval[0][0])
