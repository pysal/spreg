"""
Diagnostics for SUR and 3SLS estimation
"""

__author__ = "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com  \
             Tony Aburaad taburaad@uchicago.edu"


import numpy as np
import scipy.stats as stats
import numpy.linalg as la
from .sur_utils import sur_dict2mat, sur_mat2dict, sur_corr, spdot
from .regimes import buildR1var, wald_test


__all__ = ["sur_setp", "sur_lrtest", "sur_lmtest", "lam_setp", "surLMe", "surLMlag"]


def sur_setp(bigB, varb):
    """
    Utility to compute standard error, t and p-value

    Parameters
    ----------
    bigB    : dictionary
              of regression coefficient estimates,
              one vector by equation
    varb    : array
              variance-covariance matrix of coefficients

    Returns
    -------
    surinfdict : dictionary
                 with standard error, t-value, and
                 p-value array, one for each equation

    """
    vvb = varb.diagonal()
    n_eq = len(bigB.keys())
    bigK = np.zeros((n_eq, 1), dtype=np.int_)
    for r in range(n_eq):
        bigK[r] = bigB[r].shape[0]
    b = sur_dict2mat(bigB)
    se = np.sqrt(vvb)
    se.resize(len(se), 1)
    t = np.divide(b, se)
    tp = stats.norm.sf(abs(t)) * 2
    surinf = np.hstack((se, t, tp))
    surinfdict = sur_mat2dict(surinf, bigK)
    return surinfdict


def lam_setp(lam, vm):
    """
    Standard errors, t-test and p-value for lambda in SUR Error ML

    Parameters
    ----------
    lam        : array
                 n_eq x 1 array with ML estimates for spatial error
                 autoregressive coefficient
    vm         : array
                 n_eq x n_eq subset of variance-covariance matrix for
                 lambda and Sigma in SUR Error ML
                 (needs to be subset from full vm)

    Returns
    -------
               : tuple
                 with arrays for standard error, t-value and p-value
                 (each element in the tuple is an n_eq x 1 array)

    """
    vvb = vm.diagonal()
    se = np.sqrt(vvb)
    se.resize(len(se), 1)
    t = np.divide(lam, se)
    tp = stats.norm.sf(abs(t)) * 2
    return (se, t, tp)


def sur_lrtest(n, n_eq, ldetS0, ldetS1):
    """
    Likelihood Ratio test on off-diagonal elements of Sigma

    Parameters
    ----------
    n        : int
               cross-sectional dimension (number of observations for an equation)
    n_eq     : int
               number of equations
    ldetS0   : float
               log determinant of Sigma for OLS case
    ldetS1   : float
               log determinant of Sigma for SUR case (should be iterated)

    Returns
    -------
    (lrtest,M,pvalue) : tuple
                        with value of test statistic (lrtest),
                        degrees of freedom (M, as an integer)
                        p-value

    """
    M = n_eq * (n_eq - 1) / 2.0
    lrtest = n * (ldetS0 - ldetS1)
    pvalue = stats.chi2.sf(lrtest, M)
    return (lrtest, int(M), pvalue)


def sur_lmtest(n, n_eq, sig):
    """
    Lagrange Multiplier test on off-diagonal elements of Sigma

    Parameters
    ----------
    n        : int
               cross-sectional dimension (number of observations for an equation)
    n_eq     : int
               number of equations
    sig      : array
               inter-equation covariance matrix for null model (OLS)

    Returns
    -------
    (lmtest,M,pvalue) : tuple
                        with value of test statistic (lmtest),
                        degrees of freedom (M, as an integer)
                        p-value
    """
    R = sur_corr(sig)
    tr = np.trace(np.dot(R.T, R))
    M = n_eq * (n_eq - 1) / 2.0
    lmtest = (n / 2.0) * (tr - n_eq)
    pvalue = stats.chi2.sf(lmtest, M)
    return (lmtest, int(M), pvalue)


def surLMe(n_eq, WS, bigE, sig):
    """
    Lagrange Multiplier test on error spatial autocorrelation in SUR

    Parameters
    ----------
    n_eq       : int
                 number of equations
    WS         : array
                 spatial weights matrix in sparse form
    bigE       : array
                 n x n_eq matrix of residuals by equation
    sig        : array
                 cross-equation error covariance matrix

    Returns
    -------
    (LMe,n_eq,pvalue) : tuple
                        with value of statistic (LMe), degrees
                        of freedom (n_eq) and p-value

    """
    # spatially lagged residuals
    WbigE = WS * bigE
    # score
    EWE = np.dot(bigE.T, WbigE)
    sigi = la.inv(sig)
    SEWE = sigi * EWE
    # score = SEWE.sum(axis=1)
    # score.resize(n_eq,1)
    # note score is column sum of Sig_i * E'WE, a 1 by n_eq row vector
    # previously stored as column
    score = SEWE.sum(axis=0)
    score.resize(1, n_eq)

    # trace terms
    WW = WS * WS
    trWW = np.sum(WW.diagonal())
    WTW = WS.T * WS
    trWtW = np.sum(WTW.diagonal())
    # denominator
    SiS = sigi * sig
    Tii = trWW * np.identity(n_eq)
    tSiS = trWtW * SiS
    denom = Tii + tSiS
    idenom = la.inv(denom)
    # test statistic
    # LMe = np.dot(np.dot(score.T,idenom),score)[0][0]
    # score is now row vector
    LMe = np.dot(np.dot(score, idenom), score.T)[0][0]
    pvalue = stats.chi2.sf(LMe, n_eq)
    return (LMe, n_eq, pvalue)


def surLMlag(n_eq, WS, bigy, bigX, bigE, bigYP, sig, varb):
    """
    Lagrange Multiplier test on lag spatial autocorrelation in SUR

    Parameters
    ----------
    n_eq       : int
                 number of equations
    WS         : spatial weights matrix in sparse form
    bigy       : dictionary
                 with y values
    bigX       : dictionary
                 with X values
    bigE       : array
                 n x n_eq matrix of residuals by equation
    bigYP      : array
                 n x n_eq matrix of predicted values by equation
    sig        : array
                 cross-equation error covariance matrix
    varb       : array
                 variance-covariance matrix for b coefficients (inverse of Ibb)

    Returns
    -------
    (LMlag,n_eq,pvalue) : tuple
                          with value of statistic (LMlag), degrees
                          of freedom (n_eq) and p-value

    """
    # Score
    Y = np.hstack((bigy[r]) for r in range(n_eq))
    WY = WS * Y
    EWY = np.dot(bigE.T, WY)
    sigi = la.inv(sig)
    SEWE = sigi * EWY
    score = SEWE.sum(axis=0)  # column sums
    score.resize(1, n_eq)  # score as a row vector

    # I(rho,rho) as partitioned inverse, eq 72
    # trace terms
    WW = WS * WS
    trWW = np.sum(WW.diagonal())  # T1
    WTW = WS.T * WS
    trWtW = np.sum(WTW.diagonal())  # T2

    # I(rho,rho)
    SiS = sigi * sig
    Tii = trWW * np.identity(n_eq)  # T1It
    tSiS = trWtW * SiS
    firstHalf = Tii + tSiS
    WbigYP = WS * bigYP
    inner = np.dot(WbigYP.T, WbigYP)
    secondHalf = sigi * inner
    Ipp = firstHalf + secondHalf  # eq. 75

    # I(b,b) inverse is varb

    # I(b,rho)
    bp = sigi[0,] * spdot(
        bigX[0].T, WbigYP
    )  # initialize
    for r in range(1, n_eq):
        bpwork = (
            sigi[
                r,
            ]
            * spdot(bigX[r].T, WbigYP)
        )
        bp = np.vstack((bp, bpwork))
    # partitioned part
    i_inner = Ipp - np.dot(np.dot(bp.T, varb), bp)
    # partitioned inverse of information matrix
    Ippi = la.inv(i_inner)

    # test statistic
    LMlag = np.dot(np.dot(score, Ippi), score.T)[0][0]
    # p-value
    pvalue = stats.chi2.sf(LMlag, n_eq)
    return (LMlag, n_eq, pvalue)


def sur_chow(n_eq, bigK, bSUR, varb):
    """
    test on constancy of regression coefficients across equations in
    a SUR specification

    Note: requires a previous check on constancy of number of coefficients
    across equations; no other checks are carried out, so it is possible
    that the results are meaningless if the variables are not listed in
    the same order in each equation.

    Parameters
    ----------
    n_eq       : int
                 number of equations
    bigK       : array
                 with the number of variables by equation (includes constant)
    bSUR       : dictionary
                 with the SUR regression coefficients by equation
    varb       : array
                 the variance-covariance matrix for the SUR regression
                 coefficients

    Returns
    -------
    test       : array
                 a list with for each coefficient (in order) a tuple with the
                 value of the test statistic, the degrees of freedom, and the
                 p-value

    """
    kr = bigK[0][0]
    test = []
    bb = sur_dict2mat(bSUR)
    kf = 0
    nr = n_eq
    df = n_eq - 1
    for i in range(kr):
        Ri = buildR1var(i, kr, kf, 0, nr)
        tt, p = wald_test(bb, Ri, np.zeros((df, 1)), varb)
        test.append((tt, df, p))
    return test


def sur_joinrho(n_eq, bigK, bSUR, varb):
    """
    Test on joint significance of spatial autoregressive coefficient in SUR

    Parameters
    ----------
    n_eq       : int
                 number of equations
    bigK       : array
                 n_eq x 1 array with number of variables by equation
                 (includes constant term, exogenous and endogeneous and
                 spatial lag)
    bSUR       : dictionary
                 with regression coefficients by equation, with
                 the spatial autoregressive term as last
    varb       : array
                 variance-covariance matrix for regression coefficients

    Returns
    -------
               : tuple
                 with test statistic, degrees of freedom, p-value

    """
    bb = sur_dict2mat(bSUR)
    R = np.zeros((n_eq, varb.shape[0]))
    q = np.zeros((n_eq, 1))
    kc = -1
    for i in range(n_eq):
        kc = kc + bigK[i]
        R[i, kc] = 1
    w, p = wald_test(bb, R, q, varb)
    return (w, n_eq, p)
