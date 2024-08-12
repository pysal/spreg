"""
Spatial Fixed Effects Panel model based on: :cite:`Elhorst2003`
"""

__author__ = "Wei Kang weikang9009@gmail.com, \
              Pedro Amaral pedroamaral@cedeplar.ufmg.br, \
              Pablo Estrada pabloestradace@gmail.com"

import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu as SuperLU
from .utils import RegressionPropsY, RegressionPropsVM, inverse_prod, set_warn
from .sputils import spdot, spfill_diagonal, spinv
from . import diagnostics as DIAG
from . import user_output as USER
from . import summary_output as SUMMARY

try:
    from scipy.optimize import minimize_scalar

    minimize_scalar_available = True
except ImportError:
    minimize_scalar_available = False

from .panel_utils import check_panel, demean_panel

__all__ = ["Panel_FE_Lag", "Panel_FE_Error"]


class BasePanel_FE_Lag(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a fixed effects spatial lag model (note no consistency
    checks, diagnostics or constants added) :cite:`Elhorst2003`.

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
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   (k+1)x1 array of estimated coefficients (rho last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the rho)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1)
    vm1          : array
                   Variance covariance matrix (k+2 x k+2) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    """

    def __init__(self, y, x, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.n = w.n
        self.t = y.shape[0] // self.n
        self.k = x.shape[1]
        self.epsilon = epsilon
        # Demeaned variables
        self.y = demean_panel(y, self.n, self.t)
        self.x = demean_panel(x, self.n, self.t)
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sp.kron(sp.identity(self.t), Wsp, format="csr")
        # lag dependent variable
        ylag = spdot(Wsp_nt, self.y)
        # b0, b1, e0 and e1
        xtx = spdot(self.x.T, self.x)
        xtxi = la.inv(xtx)
        xty = spdot(self.x.T, self.y)
        xtyl = spdot(self.x.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = self.y - spdot(self.x, b0)
        e1 = ylag - spdot(self.x, b1)

        # concentrated Log Likelihood
        I = sp.identity(self.n)
        res = minimize_scalar(
            lag_c_loglik_sp,
            0.0,
            bounds=(-1.0, 1.0),
            args=(self.n, self.t, e0, e1, I, Wsp),
            method="bounded",
            options={"xatol": epsilon},
        )
        self.rho = res.x[0][0]

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - (self.n * self.t) / 2.0 * ln2pi - (self.n * self.t) / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values
        b = b0 - self.rho * b1
        self.betas = np.vstack((b, self.rho))  # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u

        xb = spdot(self.x, b)

        self.predy_e = inverse_prod(
            Wsp_nt, xb, self.rho, inv_method="power_exp", threshold=epsilon
        )
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = spdot(self.u.T, self.u) / (self.n * self.t)

        # information matrix
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(Wsp, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        wai_nt = sp.kron(sp.identity(self.t), wai, format="csr")
        wpredy = spdot(wai_nt, xb)
        xTwpy = spdot(x.T, wpredy)

        waiTwai_nt = sp.kron(sp.identity(self.t), waiTwai, format="csr")
        wTwpredy = spdot(waiTwai_nt, xb)
        wpyTwpy = spdot(xb.T, wTwpredy)

        # order of variables is beta, rho, sigma2
        v1 = np.vstack((xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
        v2 = np.vstack(
            (
                xTwpy / self.sig2,
                self.t * (tr2 + tr3) + wpyTwpy / self.sig2,
                self.t * tr1 / self.sig2,
            )
        )
        v3 = np.vstack(
            (
                np.zeros((self.k, 1)),
                self.t * tr1 / self.sig2,
                self.n * self.t / (2.0 * self.sig2 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients only
        self.varb = la.inv(np.hstack((v1[:-1], v2[:-1])))
        self.n = self.n * self.t  # change the n, for degree of freedom


class Panel_FE_Lag(BasePanel_FE_Lag):

    """
    ML estimation of the fixed effects spatial lag model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object
                   Spatial weights object
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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
                   (k+1)x1 array of estimated coefficients (rho last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the rho)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
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
                   1x(k+1) array of standard errors of the betas
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

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> nat = libpysal.examples.load_example("NCOVR")
    >>> db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
    >>> nat_shp = libpysal.examples.get_path("NAT.shp")
    >>> w = libpysal.weights.Queen.from_shapefile(nat_shp)
    >>> w.transform = 'r'
    >>> name_y = ["HR70", "HR80", "HR90"]
    >>> y = np.array([db.by_col(name) for name in name_y]).T
    >>> name_x = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
    >>> x = np.array([db.by_col(name) for name in name_x]).T
    >>> fe_lag = spreg.Panel_FE_Lag(y, x, w, name_y=name_y, name_x=name_x, name_ds="NAT")
    Warning: Assuming panel is in wide format, i.e. y[:, 0] refers to T0, y[:, 1] refers to T1, etc.
    Similarly, assuming x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    >>> np.around(fe_lag.betas, decimals=4)
    array([[ 0.8006],
           [-2.6004],
           [ 0.1903]])
    """

    def __init__(
        self,
        y,
        x,
        w,
        epsilon=0.0000001,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
    ):
        n_rows = USER.check_arrays(y, x)
        x_constant, name_x, warn = USER.check_constant(x, name_x, True)
        set_warn(self, warn)
        bigy, bigx, name_y, name_x, warn = check_panel(y, x_constant, w, name_y, name_x)
        set_warn(self, warn)
        w = USER.check_weights(w, bigy, w_required=True, time=True)

        BasePanel_FE_Lag.__init__(self, bigy, bigx, w, epsilon=epsilon)
        # increase by 1 to have correct aic and sc, include rho in count
        self.k += 1
        self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG PANEL" + " - FIXED EFFECTS"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, bigx, constant=True)
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)  # rho changed to last position
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.Panel_FE_Lag(reg=self, w=w, vm=vm)


class BasePanel_FE_Error(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a fixed effects spatial error model (note no consistency
    checks, diagnostics or constants added) :cite:`Elhorst2003`.

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
    epsilon   : float
                tolerance criterion in mimimize_scalar function and
                inverse_product

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, no constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1)
    vm1          : array
                   Variance covariance matrix (k+2 x k+2) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    """

    def __init__(self, y, x, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.n = w.n
        self.t = y.shape[0] // self.n
        self.k = x.shape[1]
        self.epsilon = epsilon
        # Demeaned variables
        self.y = demean_panel(y, self.n, self.t)
        self.x = demean_panel(x, self.n, self.t)
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sp.kron(sp.identity(self.t), Wsp, format="csr")
        # lag dependent variable
        ylag = spdot(Wsp_nt, self.y)
        xlag = spdot(Wsp_nt, self.x)

        # concentrated Log Likelihood
        I = sp.identity(self.n)
        res = minimize_scalar(
            err_c_loglik_sp,
            0.0,
            bounds=(-1.0, 1.0),
            args=(self.n, self.t, self.y, ylag, self.x, xlag, I, Wsp),
            method="bounded",
            options={"xatol": epsilon},
        )
        self.lam = res.x

        # compute full log-likelihood
        ln2pi = np.log(2.0 * np.pi)
        self.logll = (
            -res.fun - (self.n * self.t) / 2.0 * ln2pi - (self.n * self.t) / 2.0
        )

        # b, residuals and predicted values
        ys = self.y - self.lam * ylag
        xs = self.x - self.lam * xlag
        xsxs = spdot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = spdot(xs.T, ys)
        b = spdot(xsxsi, xsys)

        self.betas = np.vstack((b, self.lam))

        self.u = self.y - spdot(self.x, b)
        self.predy = self.y - self.u

        # residual variance
        self.e_filtered = self.u - self.lam * spdot(Wsp_nt, self.u)
        self.sig2 = spdot(self.e_filtered.T, self.e_filtered) / (self.n * self.t)

        # variance-covariance matrix betas
        varb = self.sig2 * xsxsi

        # variance-covariance matrix lambda, sigma
        a = -self.lam * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(Wsp, ai)
        tr1 = wai.diagonal().sum()

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        v1 = np.vstack((self.t * (tr2 + tr3), self.t * tr1 / self.sig2))
        v2 = np.vstack(
            (self.t * tr1 / self.sig2, self.t * self.n / (2.0 * self.sig2 ** 2))
        )

        v = np.hstack((v1, v2))

        self.vm1 = la.inv(v)

        # create variance matrix for beta, lambda
        vv = np.hstack((varb, np.zeros((self.k, 1))))
        vv1 = np.hstack((np.zeros((1, self.k)), self.vm1[0, 0] * np.ones((1, 1))))

        self.vm = np.vstack((vv, vv1))
        self.varb = varb
        self.n = self.n * self.t


class Panel_FE_Error(BasePanel_FE_Error):

    """
    ML estimation of the fixed effects spatial error model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, no constant
    w            : pysal W object
                   Spatial weights object
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and
                   inverse_product
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
                   kx1 array of estimated coefficients
    lam          : float
                   estimate of spatial autoregressive coefficient
    u            : array
                   (nxt)x1 array of residuals
    e_filtered   : array
                   (nxt)x1 array of spatially filtered residuals
    predy        : array
                   (nxt)x1 array of predicted y values
    n            : integer
                   Total number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (no constant, excluding the lambda)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable, including the constant
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
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    utu          : float
                   Sum of squared residuals
    std_err      : array
                   1x(k+1) array of standard errors of the betas
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

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> import spreg
    >>> nat = libpysal.examples.load_example("NCOVR")
    >>> db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
    >>> nat_shp = libpysal.examples.get_path("NAT.shp")
    >>> w = libpysal.weights.Queen.from_shapefile(nat_shp)
    >>> w.transform = 'r'
    >>> name_y = ["HR70", "HR80", "HR90"]
    >>> y = np.array([db.by_col(name) for name in name_y]).T
    >>> name_x = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
    >>> x = np.array([db.by_col(name) for name in name_x]).T
    >>> fe_error = spreg.Panel_FE_Error(y, x, w, name_y=name_y, name_x=name_x, name_ds="NAT")
    Warning: Assuming panel is in wide format, i.e. y[:, 0] refers to T0, y[:, 1] refers to T1, etc.
    Similarly, assuming x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    >>> np.around(fe_error.betas, decimals=4)
    array([[ 0.8698],
           [-2.9661],
           [ 0.1943]])
    """

    def __init__(
        self,
        y,
        x,
        w,
        epsilon=0.0000001,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
    ):
        n_rows = USER.check_arrays(y, x)
        x_constant, name_x, warn = USER.check_constant(x, name_x, True)
        set_warn(self, warn)
        bigy, bigx, name_y, name_x, warn = check_panel(y, x_constant, w, name_y, name_x)
        set_warn(self, warn)
        w = USER.check_weights(w, bigy, w_required=True, time=True)

        BasePanel_FE_Error.__init__(self, bigy, bigx, w, epsilon=epsilon)
        self.title = "MAXIMUM LIKELIHOOD SPATIAL ERROR PANEL" + " - FIXED EFFECTS"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, bigx, constant=True)
        self.name_x.append("lambda")
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.Panel_FE_Error(reg=self, w=w, vm=vm)


def lag_c_loglik_sp(rho, n, t, e0, e1, I, Wsp):
    # concentrated log-lik for lag model, sparse algebra
    if isinstance(rho, np.ndarray):
        if rho.shape == (1, 1):
            rho = rho[0][0]
    er = e0 - rho * e1
    sig2 = spdot(er.T, er)
    nlsig2 = (n * t / 2.0) * np.log(sig2)
    a = I - rho * Wsp
    LU = SuperLU(a.tocsc())
    jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
    clike = nlsig2 - jacob
    return clike


def err_c_loglik_sp(lam, n, t, y, ylag, x, xlag, I, Wsp):
    # concentrated log-lik for error model, no constants, LU
    if isinstance(lam, np.ndarray):
        if lam.shape == (1, 1):
            lam = lam[0][0]
    ys = y - lam * ylag
    xs = x - lam * xlag
    ysys = np.dot(ys.T, ys)
    xsxs = np.dot(xs.T, xs)
    xsxsi = la.inv(xsxs)
    xsys = np.dot(xs.T, ys)
    x1 = np.dot(xsxsi, xsys)
    x2 = np.dot(xsys.T, x1)
    ee = ysys - x2
    sig2 = ee[0][0]
    nlsig2 = (n * t / 2.0) * np.log(sig2)
    a = I - lam * Wsp
    LU = SuperLU(a.tocsc())
    jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
    # this is the negative of the concentrated log lik for minimization
    clik = nlsig2 - jacob
    return clik


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)
