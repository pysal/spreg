"""
Utilities for SUR and 3SLS estimation
"""

__author__ = "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"

import numpy as np
import numpy.linalg as la
from .utils import spdot

__all__ = [
    "sur_dictxy",
    "sur_dictZ",
    "sur_mat2dict",
    "sur_dict2mat",
    "sur_corr",
    "sur_crossprod",
    "sur_est",
    "sur_resids",
    "filter_dict",
    "check_k",
]


def sur_dictxy(db, y_vars, x_vars, space_id=None, time_id=None):
    """
    Stack X and y matrices and variable names as dictionaries by equation

    Parameters
    ----------
    db          : data object
                  created by libpysal.io.open
    y_vars      : array
                  list of lists with variable name(s) for dependent var
                  (Note must be a list of lists, even in splm case)
    x_vars      : array
                  list of lists with variable names for explanatory vars
    space_id    : variable with space ID
                  used for splm format
    time_id     : variable with time ID
                  used for splm format

    Return
    ------
    (bigy,bigX,bigy_vars,bigX_vars)
                : tuple
                  with dictionaries for y and X, one for
                  each equation, bigy and bigX
                  Note: bigX already includes the constant term
                  dictionaries with y and X variables, by equation
                  (includes constant for X)
    """
    c = "Constant"
    if len(y_vars) > 1:  # old format
        n_eq = len(y_vars)
        try:
            y = np.array([db[name] for name in y_vars]).T
        except:
            y = np.array([db.by_col(name) for name in y_vars]).T
        n = y.shape[0]
        bigy = {}
        bigy_vars = dict((r, y_vars[r]) for r in range(n_eq))
        bigy = dict((r, np.resize(y[:, r], (n, 1))) for r in range(n_eq))
        if not (len(x_vars) == n_eq):  # CHANGE into exception
            print("Error: mismatch variable lists")
        bigX = {}
        bigX_vars = {}
        for r in range(n_eq):
            try:
                litx = np.array([db[name] for name in x_vars[r]]).T
            except:
                litx = np.array([db.by_col(name) for name in x_vars[r]]).T
            ic = c + "_" + str(r + 1)
            x_vars[r].insert(0, ic)
            litxc = np.hstack((np.ones((n, 1)), litx))
            bigX[r] = litxc
            bigX_vars[r] = x_vars[r]
            k = litxc.shape[1]
        return (bigy, bigX, bigy_vars, bigX_vars)
    elif len(y_vars) == 1:  # splm format
        if not (time_id):  # CHANGE into exception
            print("Error: time id must be specified")
        try:
            y = np.array([db[name] for name in y_vars]).T
        except:
            y = np.array([db.by_col(name) for name in y_vars]).T
        bign = y.shape[0]
        try:
            ss = np.array([db[name] for name in space_id]).T
        except:
            ss = np.array([db.by_col(name) for name in space_id]).T
        try:
            tt = np.array([db[name] for name in time_id]).T
        except:
            tt = np.array([db.by_col(name) for name in time_id]).T
        tt1 = set([val for sublist in tt.tolist() for val in sublist])
        n_eq = len(tt1)
        tt2 = list(tt1)
        tt2.sort()
        tt3 = [str(a) for a in tt2]  # in case of string type time_id
        n = bign / n_eq
        try:
            longx = np.array([db[name] for name in x_vars[0]]).T
        except:
            longx = np.array([db.by_col(name) for name in x_vars[0]]).T
        longxc = np.hstack((np.ones((bign, 1)), longx))
        xvars = x_vars[0][:]
        xvars.insert(0, c)

        # get unique values of space_id and time_id, since they could be
        # randomly organized in a table
        skeys = []
        sdict = {}
        tmpy = {t: {} for t in tt1}
        tmpX = {t: {} for t in tt1}
        for i in range(bign):
            sval = ss[i][0]  # e.g. FIPSNO 27077
            tval = tt[i][0]  # e.g. TIME 1960
            if tmpy.has_key(tval):
                tmpy[tval][sval] = y[i]
            if tmpX.has_key(tval):
                tmpX[tval][sval] = longxc[i]
            if not sdict.has_key(sval):
                skeys.append(sval)
                sdict[sval] = True

        bigy = {}
        bigX = {}
        bigy_vars = {}
        bigX_vars = {}
        for r in range(n_eq):
            tval = tt2[r]
            bigy[r] = np.array([tmpy[tval][v] for v in skeys])
            bigX[r] = np.array([tmpX[tval][v] for v in skeys])
            bigy_vars[r] = y_vars[0] + "_" + tt3[r]
            bigX_vars[r] = [i + "_" + tt3[r] for i in xvars]
        return (bigy, bigX, bigy_vars, bigX_vars)
    else:
        print("error message, but should never be here")


def sur_dictZ(db, z_vars, form="spreg", const=False, space_id=None, time_id=None):
    """
    Generic stack data matrices and variable names as dictionaries by equation

    Parameters
    ----------
    db          : data object
                  created by libpysal.io.open
    varnames    : array
                  list of lists with variable name(s)
                  (Note must be a list of lists, even in splm case)
    form        : string
                  format used for data set.
                  default="spreg": cross-sectional format;
                  form="plm"     : plm (R) compatible using space and time id.
    const       : boolean
                  flag for constant term, default = "False"
    space_id    : variable with space ID
                  used for plm format
    time_id     : variable with time ID
                  used for plm format

    Return
    ------
    (bigZ,bigZ_names) : tuple
                        with dictionaries variables and variable
                        names, one for each equation
                        Note: bigX already includes the constant term

    """
    c = "Constant"
    if form == "spreg":  # old format
        n_eq = len(z_vars)
        bigZ = {}
        bigZ_names = {}
        for r in range(n_eq):
            try:
                litz = np.array([db[name] for name in z_vars[r]]).T
            except:
                litz = np.array([db.by_col(name) for name in z_vars[r]]).T
            if const:
                ic = c + "_" + str(r + 1)
                z_vars[r].insert(0, ic)
                litz = np.hstack((np.ones((litz.shape[0], 1)), litz))
            bigZ[r] = litz
            bigZ_names[r] = z_vars[r]
        return (bigZ, bigZ_names)

    elif form == "plm":  # plm format
        if not (time_id):  # CHANGE into exception
            raise Exception("Error: time id must be specified for plm format")
        try:
            ss = np.array([db[name] for name in space_id]).T
        except:
            ss = np.array([db.by_col(name) for name in space_id]).T
        try:
            tt = np.array([db[name] for name in time_id]).T
        except:
            tt = np.array([db.by_col(name) for name in time_id]).T
        bign = tt.shape[0]
        tt1 = set([val for sublist in tt.tolist() for val in sublist])
        n_eq = len(tt1)
        tt2 = list(tt1)
        tt2.sort()
        tt3 = [str(int(a)) for a in tt2]
        n = bign / n_eq
        try:
            longz = np.array([db[name] for name in z_vars[0]]).T
        except:
            longz = np.array([db.by_col(name) for name in z_vars[0]]).T
        zvars = z_vars[0][:]
        if const:
            longz = np.hstack((np.ones((bign, 1)), longz))
            zvars.insert(0, c)

        skeys = []
        sdict = {}
        tmpz = {t: {} for t in tt1}
        for i in range(bign):
            sval = ss[i][0]  # e.g. 27077
            tval = tt[i][0]  # e.g. 1960
            if tmpz.has_key(tval):
                tmpz[tval][sval] = longz[i]
            if not sdict.has_key(sval):
                skeys.append(sval)
                sdict[sval] = True

        bigZ = {}
        bigZ_names = {}
        for r in range(n_eq):
            tval = tt2[r]
            bigZ[r] = np.array([tmpz[tval][v] for v in skeys])
            bzvars = [i + "_" + tt3[r] for i in zvars]
            bigZ_names[r] = bzvars
        return (bigZ, bigZ_names)
    else:
        raise KeyError(
            "Invalid format used for data set. form must be either "
            " 'spreg' or 'plm', and {} was provided.".format(form)
        )


def sur_mat2dict(mat, ndim):
    """
    Utility to convert a vector or matrix to a dictionary with ndim keys,
    one for each equation

    Parameters
    ----------
    mat      : array
               vector or matrix with elements to be converted
    ndim     : array
               vector with number of elements (rows) to belong to each
               dict

    Returns
    -------
    dicts    : dictionary
               with len(ndim) keys, from 0 to len(ndim)-1


    """
    kwork = np.vstack((np.zeros((1, 1), dtype=np.int_), ndim))
    dicts = {}
    ki = 0
    for r in range(1, len(kwork)):
        ki = ki + kwork[r - 1][0]
        ke = ki + kwork[r][0]
        dicts[r - 1] = mat[ki:ke, :]
    return dicts


def sur_dict2mat(dicts):
    """
    Utility to stack the elements of a dictionary of vectors

    Parameters
    ----------
    dicts    : dictionary
               dictionary of vectors or matrices with same number
               of columns (no checks yet!)

    Returns
    -------
    mat     : array
              a vector or matrix of vertically stacked vectors


    """
    n_dicts = len(dicts.keys())
    mat = np.vstack((dicts[t] for t in range(n_dicts)))
    return mat


def sur_corr(sig):
    """
    SUR error correlation matrix

    Parameters
    ----------
    sig      : array
               Sigma cross-equation covariance matrix

    Returns
    -------
    corr  : array
            correlation matrix corresponding to sig

    """
    v = sig.diagonal()
    s = np.sqrt(v)
    s.resize(len(s), 1)
    sxs = np.dot(s, s.T)
    corr = sig / sxs
    return corr


def sur_crossprod(bigZ, bigy):
    """
    Creates dictionaries of cross products by time period for both SUR and 3SLS

    Parameters
    ----------

    bigZ       : dictionary
                 with matrix of explanatory variables,
                 including constant, exogenous and endogenous, one
                 for each equation
    bigy       : dictionary
                 with vectors of dependent variable, one
                 for each equation

    Returns
    -------
    bigZy      : dictionary
                 of all r,s cross-products of Z_r'y_s
    bigZZ      : dictionary
                 of all r,s cross-products of Z_r'Z_s
    """
    bigZZ = {}
    n_eq = len(bigy.keys())
    for r in range(n_eq):
        for t in range(n_eq):
            bigZZ[(r, t)] = spdot(bigZ[r].T, bigZ[t])
    bigZy = {}
    for r in range(n_eq):
        for t in range(n_eq):
            bigZy[(r, t)] = spdot(bigZ[r].T, bigy[t])
    return bigZZ, bigZy


def sur_est(bigXX, bigXy, bigE, bigK):
    """
    Basic SUR estimation equations for both SUR and 3SLS

    Parameters
    ----------

    bigXX        : dictionary
                   of cross-product matrices X_t'X_r
                   (created by sur_crossprod)
    bigXy        : dictionary
                   of cross-product matrices X_t'y_r
                   (created by sur_crossprod)
    bigE     : array
               n by n_eq array of residuals

    Returns
    -------

    bSUR   : dictionary
             with regression coefficients by equation
    varb   : array
             variance-covariance matrix for the regression coefficients
    sig    : array
             residual covariance matrix (using previous residuals)

    """
    n = bigE.shape[0]
    n_eq = bigE.shape[1]
    sig = np.dot(bigE.T, bigE) / n
    sigi = la.inv(sig)
    sigiXX = {}
    for r in range(n_eq):
        for t in range(n_eq):
            sigiXX[(r, t)] = bigXX[(r, t)] * sigi[r, t]
    sigiXy = {}
    for r in range(n_eq):
        sxy = 0.0
        for t in range(n_eq):
            sxy = sxy + sigi[r, t] * bigXy[(r, t)]
        sigiXy[r] = sxy
    xsigy = np.vstack((sigiXy[t] for t in range(n_eq)))
    xsigx = np.vstack(
        ((np.hstack(sigiXX[(r, t)] for t in range(n_eq))) for r in range(n_eq))
    )
    varb = la.inv(xsigx)
    beta = np.dot(varb, xsigy)
    bSUR = sur_mat2dict(beta, bigK)
    return bSUR, varb, sig


def sur_resids(bigy, bigX, beta):
    """
    Computation of a matrix with residuals by equation

    Parameters
    ----------
    bigy        : dictionary
                  with vector of dependent variable, one for each equation
    bigX        : dictionary
                  with matrix of explanatory variables, one for
                  each equation
    beta        : dictionary
                  with estimation coefficients by equation

    Returns
    -------
    bigE     : array
               a n x n_eq matrix of vectors of residuals

    """
    n_eq = len(bigy.keys())
    bigE = np.hstack((bigy[r] - spdot(bigX[r], beta[r])) for r in range(n_eq))
    return bigE


def sur_predict(bigy, bigX, beta):
    """
    Computation of a matrix with predicted values by equation

    Parameters
    ----------

    bigy        : dictionary
                  with vector of dependent variable, one for each equation
    bigX        : dictionary
                  with matrix of explanatory variables, one for
                  each equation
    beta        : dictionary
                  with estimation coefficients by equation

    Returns
    -------
    bigYP     : array
                a n x n_eq matrix of vectors of predicted values

    """
    n_eq = len(bigy.keys())
    bigYP = np.hstack(spdot(bigX[r], beta[r]) for r in range(n_eq))
    return bigYP


def filter_dict(lam, bigZ, bigZlag):
    """
    Dictionary of spatially filtered variables for use in SUR

    Parameters
    ----------
    lam        : array
                 n_eq x 1 array of spatial autoregressive parameters
    bigZ       : dictionary
                 of vector or matrix of variables, one for
                 each equation
    bigZlag    : dictionary
                 of vector or matrix of spatially lagged
                 variables, one for each equation

    Returns
    -------
    Zfilt      : dictionary
                 with spatially filtered variables
                 Z - lam*WZ, one for each equation

    """
    n_eq = lam.shape[0]
    if not (len(bigZ.keys()) == n_eq and len(bigZlag.keys()) == n_eq):
        raise Exception("Error: incompatible dimensions")
    Zfilt = {}
    for r in range(n_eq):
        lami = lam[r][0]
        Zfilt[r] = bigZ[r] - lami * bigZlag[r]
    return Zfilt


def check_k(bigK):
    """
    Check on equality of number of variables by equation

    Parameter
    ---------
    bigK     : array
               n_eq x 1 array of number of variables (includes constant)

    Returns
    -------
    True/False : result of test
    """
    kk = bigK.flatten()
    k = kk[0]
    check = np.equal(k, kk)
    return all(check)
