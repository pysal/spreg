"""
Spatial Fixed Effects Panel model based on: :cite:`KKP2007`
"""

__author__ = "Luc Anselin anselin@uchicago.edu, \
              Pedro Amaral pedroamaral@cedeplar.ufmg.br, \
              Pablo Estrada pabloestradace@gmail.com"

import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu as SuperLU
from .utils import RegressionPropsY, inverse_prod, set_warn
from .sputils import spdot, spfill_diagonal, spinv
from . import diagnostics as DIAG
from . import user_output as USER
from . import summary_output as SUMMARY
from libpysal import weights
try:
    from scipy.optimize import minimize_scalar
    minimize_scalar_available = True
except ImportError:
    minimize_scalar_available = False

from .panel_utils import check_panel, demean_panel
# import warnings

__all__ = ["Panel_ML"]


class BasePanel_ML(RegressionPropsY):

    """
    Base ML method for a fixed effects spatial lag model based on
    Anselin (2006) ...

    Parameters
    ----------
    y         : array
                (n*t)x1 array for dependent variable
    x         : array
                Two dimensional array with n*t rows and one column for each
                independent (exogenous) variable
                (note: must already include constant term)
    w         : pysal W object
                Spatial weights matrix
    method    : string
                if 'full', brute force calculation (full matrix expressions)
                if 'ord', Ord eigenvalue method
                if 'LU', LU sparse matrix decomposition
    epsilon   : float
                tolerance criterion in mimimize_scalar function and inverse_product

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    """

    def __init__(self, y, x, w,
                 approach="direct", method='full', epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.n = w.n
        self.t = y.shape[0] // self.n
        self.N = self.n * self.t
        self.k = x.shape[1]
        self.method = method
        self.epsilon = epsilon
        # Demeaned variables
        self.y = demean_panel(y, self.n, self.t)
        self.x = demean_panel(x, self.n, self.t)
        W = np.kron(np.identity(self.t), w.full()[0])
        Wsp = sp.kron(sp.identity(self.t), w.sparse)
        ylag = spdot(W, self.y)
        # b0, b1, e0 and e1
        xtx = spdot(self.x.T, self.x)
        xtxi = la.inv(xtx)
        xty = spdot(self.x.T, self.y)
        xtyl = spdot(self.x.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = self.y - spdot(self.x, b0)
        e1 = ylag - spdot(self.x, b1)
        methodML = method.upper()
        if methodML in ['FULL', 'LU', 'ORD']:
            if methodML == 'FULL':
                res = minimize_scalar(lag_c_loglik, 0.0, bounds=(-1.0, 1.0),
                                      args=(self.n, self.t, e0, e1, W),
                                      method='bounded', tol=epsilon)
            elif methodML == 'LU':
                I = sp.identity(self.N)
                res = minimize_scalar(lag_c_loglik_sp, 0.0, bounds=(-1.0, 1.0),
                                      args=(self.n, self.t, e0, e1, I, Wsp),
                                      method='bounded', tol=epsilon)
        self.rho = res.x[0][0]

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - (self.N) / 2.0 * ln2pi - self.N / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values

        b = b0 - self.rho * b1
        self.betas = np.vstack((b, self.rho))   # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u

        xb = spdot(self.x, b)

        self.predy_e = inverse_prod(
            Wsp, xb, self.rho, inv_method="power_exp", threshold=epsilon)
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = spdot(self.u.T, self.u) / self.N

        # information matrix
        # if w should be kept sparse, how can we do the following:
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        wpredy = spdot(W, self.predy_e)
        wpyTwpy = spdot(wpredy.T, wpredy)
        xTwpy = spdot(self.x.T, wpredy)

        # order of variables is beta, rho, sigma2

        v1 = np.vstack(
            (xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
        v2 = np.vstack(
            (xTwpy / self.sig2, self.t*(tr2 + tr3) + wpyTwpy / self.sig2, tr1 / self.sig2))
        v3 = np.vstack(
            (np.zeros((self.k, 1)), self.t * tr1 / self.sig2, self.n / (2.0 * self.sig2 ** 2)))

        v = np.hstack((v1, v2, v3))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients only
        self.n = self.n * self.t


class Panel_ML(BasePanel_ML):

    """
    ML estimation of the fixed effects spatial lag model with all results and
    diagnostics; :cite:`Anselin1988`

    Parameters
    ----------
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue method
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    spat_diag    : boolean
                   if True, include spatial diagnostics
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output

    Attributes
    ----------
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    method       : string
                   log Jacobian method
                   if 'full': brute force (full matrix computations)
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   Variance covariance matrix (k+2 x k+2), includes sig2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    aic          : float
                   Akaike information criterion
    schwarz      : float
                   Schwarz criterion
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
    utu          : float
                   Sum of squared residuals
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    title        : string
                   Name of the regression method used
    """

    def __init__(self, y, x, w,
                 method="full", lag=False, error="kkp", full_w=False,
                 spat_diag=False, regimes=None, vm=False,
                 name_y=None, name_x=None, epsilon=0.0000001,
                 name_w=None, name_ds=None, name_regimes=None):
        n_rows = USER.check_arrays(y, x)
        bigy, bigx, name_y, name_x = check_panel(y, x, w, name_y, name_x)
        USER.check_weights(w, bigy, w_required=True, time=True)
        # x_constant, name_x, warn = USER.check_constant(bigx, name_x)
        # set_warn(self, warn)
        method = method.upper()
        BasePanel_ML.__init__(
            self, bigy, bigx, w, method=method, epsilon=epsilon)
        self.title = "SPATIAL LAG PANEL MODEL - FIXED EFFECTS"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, bigx, constant=True)
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)  # rho changed to last position
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.Panel_ML(reg=self, w=w, vm=vm, spat_diag=spat_diag)


def lag_c_loglik(rho, n, t, e0, e1, W):
    # concentrated log-lik for lag model, no constants, brute force
    er = e0 - rho * e1
    sig2 = spdot(er.T, er) / (n*t)
    nlsig2 = (n*t / 2.0) * np.log(sig2)
    a = -rho * W
    spfill_diagonal(a, 1.0)
    jacob = t * np.log(np.linalg.det(a))
    # this is the negative of the concentrated log lik for minimization
    clik = nlsig2 - jacob
    return clik


def lag_c_loglik_sp(rho, n, t, e0, e1, I, Wsp):
    # concentrated log-lik for lag model, sparse algebra
    if isinstance(rho, np.ndarray):
        if rho.shape == (1, 1):
            rho = rho[0][0]
    er = e0 - rho * e1
    sig2 = spdot(er.T, er) / (n*t)
    nlsig2 = (n*t / 2.0) * np.log(sig2)
    a = I - rho * Wsp
    LU = SuperLU(a.tocsc())
    jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
    clike = nlsig2 - jacob
    return clike


def _test():
    import doctest
    start_suppress = np.get_printoptions()['suppress']
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


# if __name__ == '__main__':
#     _test()
