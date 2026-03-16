"""
Diagnostics for panel data estimation
"""

__author__ = "Wei Kang weikang9009@gmail.com, \
              Pedro Amaral pedroamaral@cedeplar.ufmg.br, \
              Pablo Estrada pabloestradace@gmail.com"

import numpy as np
import numpy.linalg as la
import pandas as pd
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
    y, x, name_y, name_x, T, warn = check_panel(y, x, w, None, None)
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
    y, x, name_y, name_x, T, warn = check_panel(y, x, w, None, None)
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
    y, x, name_y, name_x, T, warn = check_panel(y, x, w, None, None)
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
    y, x, name_y, name_x, T, warn = check_panel(y, x, w, None, None)
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

def BSK_tests(reg, w, which="all", GM_BSK=False):
    """
    Perform Baltagi, Song, and Koh (2003) LM tests for Spatial Panel Data.
    
    Tests Included:
    1. LM Joint: Joint test for Random Effects and Spatial Error.
    2. LM Marginal RE: Test for RE assuming no Spatial Error.
    3. LM Marginal Spatial: Test for Spatial Error assuming no RE.
    4. LM Conditional Spatial: Test for Spatial Error allowing for RE.
    5. LM Conditional RE: Test for RE allowing for Spatial Error.

    Parameters
    ----------
    reg     : spreg regression object
              A fitted spreg panel model 
              (usually either PooledOLS, PanelRE, ML_ErrorPooled, or GM_ErrorPooled)
    w       : pysal W object
              Spatial weights matrix.
    GM_BSK : boolean
              If True, use GM estimator for Pooled SEM.
              Ignored if reg is already a GM_ErrorPooled or ML_ErrorPooled instance.

    Returns
    -------
    summary : pd.DataFrame
              Table with Test Statistic, p-value, and Conclusion.
    """
    from .panel import PanelRE, BaseML_ErrorPooled, BaseGM_ErrorPooled, PooledOLS

    if which == "all":
        which = ["LMJ", "LM1", "LM2", "LMC_spatial", "LMC_RE"]
    elif isinstance(which, str):
        which = [which]

    test_names = []
    stats_list = []
    dfs = []
    pvals = []

    if hasattr(w, 'trcWtW_WW'):
        b = w.trcWtW_WW
    else:
        w_sparse = w.sparse
        wtw = w_sparse.T @ w_sparse
        w2 = w_sparse @ w_sparse
        b = (wtw + w2).diagonal().sum()    

    N = w.n
    T = reg.n // N

    if any(test in which for test in ["LMJ", "LM1", "LM2"]):
        if reg.__class__.__name__ != 'PooledOLS':
            pooled_mod = PooledOLS(reg.y, reg.x)
        else:
            pooled_mod = reg

        if any(test in which for test in ["LMJ", "LM1"]):
            # G term (Random Effects) - Eq 2.11
            u = pooled_mod.u 
            u_mat = u.reshape((T, N))
            u_sum_sq = np.sum(np.sum(u_mat, axis=0)**2) 
            u_sq_sum = np.sum(u**2)
            G = (u_sum_sq / u_sq_sum) - 1

            # LM1: Marginal RE (Eq 2.12)
            LM1 = np.sqrt( (N * T) / (2 * (T - 1)) ) * G
            pval_LM1 = 1 - stats.norm.cdf(LM1) 

            if "LM1" in which:
                test_names.append("LM Marginal RE (LM1)")
                stats_list.append(LM1)
                dfs.append(1)
                pvals.append(pval_LM1)            

        if any(test in which for test in ["LMJ", "LM2"]):
            """
            # H matrix approach - matches vectorial below (but slower)
            w_full = w.full()[0]
            w_wt = w_full + w_full.T
            w_wt2 = w_wt / 2.0 # LA
            b2 = np.trace(np.dot(w_wt, w_wt))/2.0
            #Iwwt = np.kron(np.eye(T), w_wt) / 2
            Iwwt = np.kron(np.eye(T), w_wt2) # LA
            num_H2 = np.dot(np.dot(u.T, Iwwt), u).item()
            """
            # H term (Spatial Error) - Eq 2.11
            Wu_mat = u_mat @ w.sparse.T
            Wu_vec = Wu_mat.reshape(-1, 1)
            num_H = np.sum(u.flatten() * Wu_vec.flatten())
            H = num_H / u_sq_sum

            # LM2: Marginal Spatial (Eq 2.16)
            LM2 = np.sqrt( (N**2 * T) / b ) * H
            pval_LM2 = 1 - stats.norm.cdf(LM2) 

            if "LM2" in which:
                test_names.append("LM Marginal Spatial (LM2)")
                stats_list.append(LM2)
                dfs.append(1)
                pvals.append(pval_LM2)

        if "LMJ" in which:
            # Joint Test (Eq 2.19)
            if LM1 > 0 and LM2 > 0:
                LM_J = LM1**2 + LM2**2
                prob_ch1 = 1 - stats.chi2.cdf(LM_J, df=1)
                prob_ch2 = 1 - stats.chi2.cdf(LM_J, df=2)
                pval_J = 0.5 * prob_ch1 + 0.25 * prob_ch2
            elif LM1 > 0 and LM2 <= 0:
                LM_J = LM1**2
                pval_J = 0.5 * (1 - stats.chi2.cdf(LM_J, df=1)) 
            elif LM1 <= 0 and LM2 > 0:
                LM_J = LM2**2
                pval_J = 0.5 * (1 - stats.chi2.cdf(LM_J, df=1))
            else:
                LM_J = 0
                pval_J = 1.0

            test_names.append("LM Joint (LMJ)")
            stats_list.append(LM_J)
            dfs.append(2)
            pvals.append(pval_J)

    if "LMC_spatial" in which:
        if reg.__class__.__name__ != 'PanelRE':
            re_mod = PanelRE(reg.y, reg.x, w, spat_diag=False)
        else:
            re_mod = reg
        # Conditional LM Spatial (Eq 2.24 - 2.25)
        u_re_raw = re_mod.y - np.dot(re_mod.x,re_mod.betas)
        u_re_mat = u_re_raw.reshape((T, N))
        
        """
        # Matrix computation - matches vectorial below
        JbT = np.ones((T,T)) / T
        Et = np.eye(T) - JbT
        Et_k_In = np.kron(Et, np.eye(N))
        JbT_k_In = np.kron(JbT, np.eye(N))
        sig2_v = np.dot(np.dot(u_re_raw.T, Et_k_In), u_re_raw).item() / (N * (T-1))
        sig2_1 = np.dot(np.dot(u_re_raw.T, JbT_k_In), u_re_raw).item() / N
        wtw = w_full.T + w_full
        Dp1 = (sig2_v/(sig2_1**2))*np.kron(JbT, wtw)
        Dp2 = (1/sig2_v)*np.kron(Et, wtw)
        D_lambda_alt = 0.5 * np.dot(np.dot(u_re_raw.T, Dp1 + Dp2), u_re_raw).item()
        denon_var_alt = ((T-1) + (sig2_v**2 / sig2_1**2)) * b
        """

        # Vector computation
        u_re_mean = u_re_mat.mean(axis=0) 
        sig_1sq = np.sum(T * u_re_mean**2) / N
        
        u_dev = u_re_mat - u_re_mean 
        sig_v2 = np.sum(u_dev**2) / (N * (T - 1))
        
        # Term 1:
        Wu_mean = w.sparse.dot(u_re_mean)
        quad_mean = np.sum(u_re_mean * Wu_mean) * 2 
        term1 = (sig_v2 / (sig_1sq**2)) * T * quad_mean
        
        # Term 2:
        Wu_re_mat = u_re_mat @ w.sparse.T
        quad_full = np.sum(u_re_mat * Wu_re_mat) * 2
        term2 = (1 / sig_v2) * (quad_full - (T * quad_mean))
        
        D_lambda = 0.5 * (term1 + term2)
        denom_var = ( (T-1) + (sig_v2**2 / sig_1sq**2) ) * b
        LM_cond_spatial = D_lambda / np.sqrt(denom_var)
        pval_cond_spatial = 1 - stats.norm.cdf(LM_cond_spatial)

        test_names.append("LM Conditional Spatial")
        stats_list.append(LM_cond_spatial)
        dfs.append(1)
        pvals.append(pval_cond_spatial)

    if "LMC_RE" in which:
        # Conditional LM RE (Eq 2.26 - 2.31)
        if reg.__class__.__name__ not in ('BaseGM_ErrorPooled', 'GM_ErrorPooled', 'BaseML_ErrorPooled', 'ML_ErrorPooled'):
            if GM_BSK:
                sem_mod = BaseGM_ErrorPooled(reg.y, reg.x, w)
            else:
                sem_mod = BaseML_ErrorPooled(reg.y, reg.x, w)
        else:
            sem_mod = reg
        """
        # Matrix computation (slower)
        sem_mod = BaseML_ErrorPooled(reg_ols.y, reg_ols.x, w)
        lam_est = sem_mod.betas[-1,0]
        u_sem = sem_mod.u

        W_mat = w.full()[0]
        B_est = np.eye(N) - lam_est * W_mat
        BtB = np.dot(B_est.T, B_est)
        
        sig2_v = np.dot(np.dot(u_sem.T,np.kron(np.eye(T),BtB)),u_sem).item() / (N*T) # Eq. A.29
        sig4_v = sig2_v**2

        BtB2 = np.dot(BtB, BtB)
        JTBB2 = np.kron(np.ones((T,T)), BtB2)
        D_mu_p2 = np.dot(np.dot(u_sem.T,JTBB2), u_sem).item() / (2*sig4_v)
        D_mu_alt = D_mu_p2 - (np.trace(BtB) * (T/(2*sig2_v)))

        h = np.trace(BtB)
        WtB_BtW = np.dot(W_mat.T, B_est) + np.dot(B_est.T, W_mat) 
        d = np.trace(WtB_BtW)
        e = np.trace(BtB2)
        BtB_inv = np.linalg.inv(BtB) 
        g = np.trace(np.dot(WtB_BtW, BtB_inv))
        term_c = np.dot(WtB_BtW, BtB_inv)
        c = np.trace(np.dot(term_c, term_c))

        LM_mu_num = D_mu_alt * np.sqrt((2*sig4_v/T) * ((N*sig4_v*c) - (sig4_v*(g**2))))
        LM_mu_den = np.sqrt((T*N*sig4_v*e*c)-(N*sig4_v*(d**2))-(T*sig4_v*e*(g**2))+(2*sig4_v*g*h*d)-(sig4_v*c*(h**2)))
        LM_mu_alt = LM_mu_num / LM_mu_den
        """
    
        # Vector approach
        lam_est = sem_mod.betas[-1,0]
        u_sem = sem_mod.u 
        u_sem_mat = u_sem.reshape((T, N))

        W_mat = w.full()[0]
        B_est = np.eye(N) - lam_est * W_mat
        BB = np.dot(B_est.T, B_est)
        BB_inv = np.linalg.inv(BB) 
        
        v_mat = np.zeros_like(u_sem_mat)
        for t in range(T):
            v_mat[t, :] = B_est @ u_sem_mat[t, :]
        v = v_mat.reshape(-1, 1)
        
        sig2_v = (v.T @ v) / (N * T)
        sig2_v = sig2_v[0,0]
        
        WB_BW = np.dot(W_mat.T, B_est) + np.dot(B_est.T, W_mat) 
        
        h = np.trace(BB)
        d = np.trace(WB_BW)
        e = np.trace(np.dot(BB, BB))
        g = np.trace(np.dot(WB_BW, BB_inv))
        
        term_c = np.dot(WB_BW, BB_inv)
        c = np.trace(np.dot(term_c, term_c))
        
        t1 = - (T / (2 * sig2_v)) * h
        
        u_mean = u_sem_mat.mean(axis=0)
        quad_form = u_mean.T @ np.dot(BB, BB) @ u_mean
        t2 = (1 / (2 * sig2_v**2)) * (T**2) * quad_form
        D_mu = t1 + t2
        
        term_inside_num = (2/T) * (N*c - g**2)
        if term_inside_num < 0: term_inside_num = 0
        numerator_final = D_mu * sig2_v**2 * np.sqrt(term_inside_num)
        
        denom_val = T*N*e*c - N*d**2 - T*g**2*e + 2*g*h*d - h**2*c
        if denom_val < 0: denom_val = 0
        denominator_final = sig2_v * np.sqrt(denom_val)
        
        if denominator_final == 0:
            LM_cond_RE = 0
        else:
            LM_cond_RE = numerator_final / denominator_final
            
        pval_cond_RE = 1 - stats.norm.cdf(LM_cond_RE)

        test_names.append("LM Conditional RE")
        stats_list.append(LM_cond_RE)
        dfs.append(1)
        pvals.append(pval_cond_RE)

    res_data = {
        "Test": test_names,
        "Statistic": stats_list,
        "df": dfs,
        "p-value": pvals
    }
    
    return pd.DataFrame(res_data)