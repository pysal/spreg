"""
Utilities for panel data estimation
"""

__author__ = "Wei Kang weikang9009@gmail.com, \
              Pedro Amaral pedroamaral@cedeplar.ufmg.br, \
              Pablo Estrada pabloestradace@gmail.com"

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.linalg import solve_triangular
from .sputils import spdot
from . import user_output as USER


def _error_re_filter(y, x, W, t, lam, ratio):
    """Whiten SEM RE data with iid individual effects; ratio = sigma2_u / sigma2.

    For B = I - lambda W, the within covariance is (B'B)^-1 and the
    between covariance is (B'B)^-1 + T * ratio * I. Factoring the latter
    after applying B avoids diagonalizing a possibly nonsymmetric W.
    Returns filtered y, filtered X, and log|Omega / sigma2|.
    """
    n = W.shape[0]
    B = np.eye(n) - lam * W
    z = np.column_stack((y, x))
    means = z.reshape(t, n, -1).mean(axis=0)
    centered = z - np.tile(means, (t, 1))
    within = centered - lam * np.matmul(W, centered.reshape(t, n, -1)).reshape(z.shape)
    between_cov = np.eye(n) + t * ratio * (B @ B.T)
    chol = np.linalg.cholesky(between_cov)
    between = solve_triangular(chol, B @ means, lower=True, check_finite=False)
    filtered = within + np.tile(between, (t, 1))
    logdet = 2 * np.log(np.diag(chol)).sum() - 2 * t * np.linalg.slogdet(B)[1]
    return filtered[:, :1], filtered[:, 1:], logdet


def prepare_panel(y, x, w, name_y, name_x, slx_lags, slx_vars, title, time_effects, add_constant=True):
    """
    Prepare the data for panel regression.
    Parameters as in the panel regression classes, with the addition of:
    add_constant : bool
                   Whether to add a constant term to the data. Default is True.
    """
    warn = []
    n_rows = USER.check_arrays(y, x)
    x_constant, name_x, warn_add = USER.check_constant(x, name_x, just_rem=True)
    warn.append(warn_add)
    bigy, bigx, name_y, name_x, T, warn_add = check_panel(y, x_constant, w, name_y, name_x)
    warn.append(warn_add)
    if add_constant:
        bigx, name_x, warn_add = USER.check_constant(bigx, name_x)
        warn.append(warn_add)
    name_x = USER.set_name_x(name_x, bigx, constant=not add_constant)
    name_y = USER.set_name_y(name_y)

    w = USER.check_weights(w, bigy, w_required=True, time=True, slx_lags=slx_lags)

    kt = 0
    if time_effects and T > 1:
        n = w.n
        time_dummies = np.repeat(np.eye(T), n, axis=0)[:, 1:]
        time_names = [f"time_{t}" for t in range(2, T + 1)]
        bigx = np.hstack((bigx, time_dummies))
        name_x = list(name_x) + time_names
        kt = T - 1

    if slx_lags > 0 and kt > 0:
        n_work = bigx.shape[1] - (1 if add_constant else 0)  # non-constant cols
        n_orig = n_work - kt
        if isinstance(slx_vars, str) and slx_vars.lower() == "all":
            slx_vars = [True] * n_orig + [False] * kt
        elif isinstance(slx_vars, list):
            if len(slx_vars) != n_orig:
                raise ValueError(
                    f"slx_vars length ({len(slx_vars)}) must equal the number of "
                    f"exogenous variables excluding constant and time effects ({n_orig})."
                )
            slx_vars = list(slx_vars) + [False] * kt
        else:
            raise ValueError("slx_vars must be 'all' or a list of booleans.")

    kx = bigx.shape[1]

    if slx_lags > 0:
        title += " WITH SLX"
        bigx, name_x = USER.flex_wx(
            w, x=bigx, name_x=name_x, constant=add_constant,
            slx_lags=slx_lags, slx_vars=slx_vars, panel=True,
        )
    kwx = bigx.shape[1] - kx

    return bigy, bigx, name_y, name_x, w, warn, slx_lags, slx_vars, title, kx, kwx, T

def check_panel(y, x, w, name_y, name_x):
    """
    Check the data structure and converts from wide to long if needed.

    Parameters
    ----------
    y          : array
                 n*tx1 or nxt array for dependent variable
    x          : array
                 Two dimensional array with n*t rows and k columns for
                 independent (exogenous) variable or n rows and k*t columns
                 (note, must not include a constant term)
    name_y     : string or list of strings
                 Name of dependent variable for use in output
    name_x     : list of strings
                 Names of independent variables for use in output
    """

    if isinstance(y, (pd.Series, pd.DataFrame)):
        if name_y is None:
            try:
                name_y = y.columns.to_list()
            except AttributeError:
                name_y = y.name
        y = y.to_numpy()
        
    if isinstance(x, (pd.Series, pd.DataFrame)):
        if name_x is None:
            try:
                name_x = x.columns.to_list()
            except AttributeError:
                name_x = x.name
        x = x.to_numpy()

    # Check if 'y' is a balanced panel with respect to 'W'
    if y.shape[0] / w.n != y.shape[0] // w.n:
        raise Exception("y must be ntx1 or nxt, and w must be an nxn PySAL W" "object.")
    # Wide format
    if y.shape[1] > 1:
        warn = (
            "Assuming panel is in wide format.\n"
            "y[:, 0] refers to T0, y[:, 1] refers to T1, etc.\n"
            "x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers "
            "to k2, etc."
        )
        N, T = y.shape[0], y.shape[1]
        k = x.shape[1] // T
        bigy = y.reshape((y.size, 1), order="F")
        bigx = x[:, 0:T].reshape((N * T, 1), order="F")
        for i in range(1, k):
            bigx = np.hstack(
                (bigx, x[:, T * i : T * (i + 1)].reshape((N * T, 1), order="F"))
            )
    # Long format
    else:
        warn = (
            "Assuming panel is in long format.\n"
            "y[0:N] refers to T0, y[N+1:2N] refers to T1, etc.\n"
            "x[0:N] refers to T0, x[N+1:2N] refers to T1, etc."
        )
        T = y.shape[0] // w.n
        N = w.n
        k = x.shape[1]
        bigy, bigx = y, x
    # Fix name_y and name_x
    if name_y:
        if not isinstance(name_y, str) and not isinstance(name_y, list):
            raise Exception("name_y must either be strings or a list of" "strings.")
        if len(name_y) > 1 and isinstance(name_y, list):
            name_y = "".join([i for i in name_y[0] if not i.isdigit()])
        if len(name_y) == 1 and isinstance(name_y, list):
            name_y = name_y[0]
    if name_x:
        if len(name_x) != k * T and len(name_x) != k:
            raise Exception(
                "Names of columns in X must have exactly either" "k or k*t elements."
            )
        if len(name_x) > k:
            name_bigx = []
            for i in range(k):
                namek = "".join([j for j in name_x[i * T] if not j.isdigit()])
                name_bigx.append(namek)
            name_x = name_bigx

    return bigy, bigx, name_y, name_x, T, warn


def demean_panel(arr, n, t, phi=0):
    """
    Returns demeaned variable.

    Parameters
    ----------
    arr         : array
                  n*tx1 array
    n           : integer
                  Number of observations
    t           : integer
                  Number of time periods
    phi         : float
                  Weight from 0 to 1 attached to the cross-sectional component
                  of the data. If phi=0, then it is the demeaning procedure.
                  If phi=1, then the data doesn't change at all.

    Returns
    -------
    arr_dm      : array
                  Demeaned variable
    """

    one = np.ones((t, 1))
    J = sp.identity(t) - (1 - phi) * (1 / t) * spdot(one, one.T)
    Q = sp.kron(J, sp.identity(n), format="csr")
    arr_dm = spdot(Q, arr)

    return arr_dm

def time_inv_check(dem_x, name_x):
    """
    Check for time-invariant variables in a demeaned matrix.
    
    Parameters
    ----------
    dem_x  : array
             Demeaned independent variables array (n*t x k)
    name_x : list of strings
             Names of independent variables
             
    Raises
    ------
    ValueError
        If time-invariant variables are detected (variance is zero).
    """
    var_check = np.var(dem_x, axis=0)
    invalid_cols = np.where(var_check <= 1e-12)[0]
    
    if invalid_cols.size > 0:
        invalid_names = [name_x[i] for i in invalid_cols]
        raise ValueError(
            f"Time-invariant variables detected: {', '.join(invalid_names)}. "
            "These cannot be estimated using the Within transformation. "
            "Please remove them from the input variables (x)."
        )