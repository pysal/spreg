"""
Spatial Random Effects Panel model based on: :cite:`Elhorst2003`
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
from spreg.w_utils import symmetrize
from . import diagnostics as DIAG
from . import user_output as USER
from . import summary_output as SUMMARY

try:
    from scipy.optimize import minimize_scalar

    minimize_scalar_available = True
except ImportError:
    minimize_scalar_available = False
try:
    from scipy.optimize import minimize

    minimize_available = True
except ImportError:
    minimize_available = False

from .panel_utils import check_panel, demean_panel

__all__ = ["Panel_RE_Lag", "Panel_RE_Error"]


class BasePanel_RE_Lag(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a random effects spatial lag model (note no consistency
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
                   (k+2)x1 array of estimated coefficients (rho and phi last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    phi          : float
                   estimate of weight attached to the cross-sectional component
                   phi^2 = sig2 / (t*sig2_u + sig2)
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
                   (no constant, excluding the rho and phi)
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
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    """

    def __init__(self, bigy, bigx, w, epsilon=0.0000001):
        # set up main regression variables and spatial filters
        self.n = w.n
        self.t = bigy.shape[0] // self.n
        self.k = bigx.shape[1]
        self.epsilon = epsilon
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sp.kron(sp.identity(self.t), Wsp, format="csr")
        # Set up parameters
        converge = 1
        criteria = 0.0000001
        i = 0
        itermax = 100
        self.rho = 0.1
        self.phi = 0.1
        I = sp.identity(self.n)
        xtx = spdot(bigx.T, bigx)
        xtxi = la.inv(xtx)
        xty = spdot(bigx.T, bigy)
        b = spdot(xtxi, xty)

        # Iterative procedure
        while converge > criteria and i < itermax:
            phiold = self.phi
            res_phi = minimize_scalar(
                phi_c_loglik,
                0.1,
                bounds=(0.0, 1.0),
                args=(self.rho, b, bigy, bigx, self.n, self.t, Wsp_nt),
                method="bounded",
                options={"xatol": epsilon},
            )
            self.phi = res_phi.x[0][0]
            # Demeaned variables
            self.y = demean_panel(bigy, self.n, self.t, phi=self.phi)
            self.x = demean_panel(bigx, self.n, self.t, phi=self.phi)
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
            res_rho = minimize_scalar(
                lag_c_loglik_sp,
                0.0,
                bounds=(-1.0, 1.0),
                args=(self.n, self.t, e0, e1, I, Wsp),
                method="bounded",
                options={"xatol": epsilon},
            )
            self.rho = res_rho.x[0][0]
            b = b0 - self.rho * b1
            i += 1
            converge = np.abs(phiold - self.phi)

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res_rho.fun - (self.n * self.t) / 2.0 * ln2pi - (self.n * self.t) / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values
        self.betas = np.vstack((b, self.rho, self.phi))
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
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        wai_nt = sp.kron(sp.identity(self.t), wai, format="csr")
        wpredy = spdot(wai_nt, xb)
        xTwpy = spdot(self.x.T, wpredy)

        waiTwai_nt = sp.kron(sp.identity(self.t), waiTwai, format="csr")
        wTwpredy = spdot(waiTwai_nt, xb)
        wpyTwpy = spdot(xb.T, wTwpredy)

        # order of variables is beta, rho, sigma2
        v1 = np.vstack((xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((2, self.k))))
        v2 = np.vstack(
            (
                xTwpy / self.sig2,
                self.t * (tr2 + tr3) + wpyTwpy / self.sig2,
                -tr1 / self.sig2,
                self.t * tr1 / self.sig2,
            )
        )
        v3 = np.vstack(
            (
                np.zeros((self.k, 1)),
                -tr1 / self.sig2,
                self.n * (1 + 1 / self.phi ** 2),
                -self.n / self.sig2,
            )
        )
        v4 = np.vstack(
            (
                np.zeros((self.k, 1)),
                self.t * tr1 / self.sig2,
                -self.n / self.sig2 ** 2,
                self.n * self.t / (2.0 * self.sig2 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3, v4))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients and phi
        self.varb = la.inv(np.hstack((v1[:-2], v2[:-2])))
        self.n = self.n * self.t


class Panel_RE_Lag(BasePanel_RE_Lag):

    """
    ML estimation of the random effects spatial lag model with all results and
    diagnostics :cite:`Elhorst2003`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas object
                   nxt or (nxt)x1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   nx(txk) or (nxt)xk array for independent (exogenous)
                   variables, excluding the constant
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
                   (k+2)x1 array of estimated coefficients (rho and phi last)
    rho          : float
                   estimate of spatial autoregressive coefficient
    phi          : float
                   estimate of weight attached to the cross-sectional component
                   phi^2 = sig2 / (t*sig2_u + sig2)
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
                   (excluding the phi)
    y            : array
                   (nxt)x1 array for dependent variable
    x            : array
                   Two dimensional array with nxt rows and one column for each
                   independent (exogenous) variable
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
    sig2         : float
                   Sigma squared used in computations
    logll        : float
                   maximized log-likelihood (including constant terms)
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
    >>> import pandas as pd
    >>> import libpysal
    >>> import spreg
    >>> nat = libpysal.examples.load_example("NCOVR")
    >>> db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
    >>> nat_shp = libpysal.examples.get_path("NAT.shp")
    >>> w_full = libpysal.weights.Queen.from_shapefile(nat_shp)
    >>> name_y = ["HR70", "HR80", "HR90"]
    >>> y_full = np.array([db.by_col(name) for name in name_y]).T
    >>> name_x = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
    >>> x_full = np.array([db.by_col(name) for name in name_x]).T
    >>> name_c = ["STATE_NAME", "FIPSNO"]
    >>> df_counties = pd.DataFrame([db.by_col(name) for name in name_c], index=name_c).T
    >>> filter_states = ["Kansas", "Missouri", "Oklahoma", "Arkansas"]
    >>> filter_counties = df_counties[df_counties["STATE_NAME"].isin(filter_states)]["FIPSNO"].values
    >>> counties = np.array(db.by_col("FIPSNO"))
    >>> subid = np.where(np.isin(counties, filter_counties))[0]
    >>> w = w_subset(w_full, subid)
    >>> w.transform = 'r'
    >>> y = y_full[subid, ]
    >>> x = x_full[subid, ]
    >>> re_lag = spreg.Panel_RE_Lag(y, x, w, name_y=name_y, name_x=name_x, name_ds="NAT")
    Warning: Assuming panel is in wide format, i.e. y[:, 0] refers to T0, y[:, 1] refers to T1, etc.
    Similarly, assuming x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    np.around(re_lag.betas, decimals=4)
    array([[4.44421994],
           [2.52821717],
           [2.24768846],
           [0.25846846],
           [0.68426639]])
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
        bigy, bigx, name_y, name_x, warn = check_panel(y, x, w, name_y, name_x)
        set_warn(self, warn)
        bigx, name_x, warn = USER.check_constant(bigx, name_x)
        set_warn(self, warn)
        w = USER.check_weights(w, bigy, w_required=True, time=True)

        BasePanel_RE_Lag.__init__(self, bigy, bigx, w, epsilon=epsilon)
        # increase by 1 to have correct aic and sc, include rho in count
        self.k += 1
        self.title = "MAXIMUM LIKELIHOOD SPATIAL LAG PANEL" + " - RANDOM EFFECTS"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, bigx, constant=False)
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)  # rho changed to last position
        self.name_x.append("phi")  # error variance parameter
        self.name_w = USER.set_name_w(name_w, w)
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.Panel_FE_Lag(reg=self, w=w, vm=vm)


class BasePanel_RE_Error(RegressionPropsY, RegressionPropsVM):

    """
    Base ML method for a random effects spatial error model (note no
    consistency checks, diagnostics or constants added) :cite:`Elhorst2003`.

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
    sig2_u       : float
                   Sigma squared for random effects
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
                   Variance covariance matrix (k+2 x k+2)
    vm1          : array
                   Variance covariance matrix (k+3 x k+3) includes sigma2
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
        self.y = y
        self.x = x
        # Big W matrix
        W = w.full()[0]
        Wsp = w.sparse
        Wsp_nt = sp.kron(sp.identity(self.t), Wsp, format="csr")
        # lag dependent variable
        ylag = spdot(Wsp_nt, self.y)
        xlag = spdot(Wsp_nt, self.x)

        # concentrated Log Likelihood
        I = np.identity(self.n)
        if w.asymmetry(intrinsic=False) == []:
            ww = symmetrize(w)
            WW = np.array(ww.todense())
            evals, evecs = la.eigh(WW)
            W = WW
        else:  # need dense here
            evals, evecs = la.eig(W)
        one = np.ones((self.t, 1))
        J = (1 / self.t) * spdot(one, one.T)
        Q = sp.kron(J, I, format="csr")
        y_mean = spdot(Q, self.y)
        x_mean = spdot(Q, self.x)
        res = minimize(
            err_c_loglik_ord,
            (0.0, 0.1),
            bounds=((-1.0, 1.0), (0.0, 10000.0)),
            method="L-BFGS-B",
            args=(
                evals,
                evecs,
                self.n,
                self.t,
                self.y,
                self.x,
                ylag,
                xlag,
                y_mean,
                x_mean,
                I,
                Wsp,
            ),
        )
        self.lam, self.phi = res.x

        # compute full log-likelihood
        ln2pi = np.log(2.0 * np.pi)
        self.logll = (
            -res.fun - (self.n * self.t) / 2.0 * ln2pi - (self.n * self.t) / 2.0
        )

        # b, residuals and predicted values
        cvals = self.t * self.phi ** 2 + 1 / (1 - self.lam * evals) ** 2
        P = spdot(np.diag(cvals ** (-0.5)), evecs.T)
        pr = P - (I - self.lam * W)
        pr_nt = sp.kron(sp.identity(self.t), pr, format="csr")
        yrand = self.y + spdot(pr_nt, y_mean)
        xrand = self.x + spdot(pr_nt, x_mean)
        ys = yrand - self.lam * ylag
        xs = xrand - self.lam * xlag
        xsxs = spdot(xs.T, xs)
        xsxsi = la.inv(xsxs)
        xsys = spdot(xs.T, ys)
        b = spdot(xsxsi, xsys)

        self.u = self.y - spdot(self.x, b)
        self.predy = self.y - self.u

        # residual variance
        self.e_filtered = ys - spdot(xs, b)
        self.sig2 = spdot(self.e_filtered.T, self.e_filtered) / (self.n * self.t)

        # variance-covariance matrix betas
        varb = self.sig2 * xsxsi
        # variance of random effects
        self.sig2_u = self.phi ** 2 * self.sig2

        self.betas = np.vstack((b, self.lam, self.sig2_u))

        # variance-covariance matrix lambda, sigma
        a = -self.lam * W
        spfill_diagonal(a, 1.0)
        aTai = la.inv(spdot(a.T, a))
        wa_aw = spdot(W.T, a) + spdot(a.T, W)
        gamma = spdot(wa_aw, aTai)
        vi = la.inv(self.t * self.phi * I + aTai)
        sigma = spdot(vi, aTai)

        tr1 = gamma.diagonal().sum()
        tr2 = vi.diagonal().sum()
        tr3 = sigma.diagonal().sum()

        sigma_gamma = spdot(sigma, gamma)
        tr4 = sigma_gamma.diagonal().sum()

        sigma_vi = spdot(sigma, vi)
        tr5 = sigma_vi.diagonal().sum()

        sigma_gamma_vi = spdot(sigma_gamma, vi)
        tr6 = sigma_gamma_vi.diagonal().sum()

        sigma_gamma_sigma = spdot(sigma_gamma, sigma)
        tr7 = sigma_gamma_sigma.diagonal().sum()

        v1 = np.vstack(
            (
                (self.t - 1) / 2 * tr1 ** 2 + 1 / 2 * tr4 ** 2,
                self.t / (2 * self.sig2) * tr6,
                (self.t - 1) / (2 * self.sig2) * tr1 + 1 / (2 * self.sig2) * tr7,
            )
        )
        v2 = np.vstack(
            (
                self.t / (2 * self.sig2) * tr6,
                self.t ** 2 / (2.0 * self.sig2 ** 2) * tr2 ** 2,
                self.t / (2.0 * self.sig2 ** 2) * tr5,
            )
        )
        v3 = np.vstack(
            (
                (self.t - 1) / (2 * self.sig2) * tr1 + 1 / (2 * self.sig2) * tr7,
                self.t / (2.0 * self.sig2 ** 2) * tr5,
                1 / (2.0 * self.sig2 ** 2) * ((self.t - 1) * self.n + tr3 ** 2),
            )
        )

        v = np.hstack((v1, v2, v3))

        vm1 = np.linalg.inv(v)

        # create variance matrix for beta, lambda
        vv = np.hstack((varb, np.zeros((self.k, 2))))
        vv1 = np.hstack((np.zeros((2, self.k)), vm1[:2, :2]))

        self.vm = np.vstack((vv, vv1))
        self.varb = varb
        self.n = self.n * self.t


class Panel_RE_Error(BasePanel_RE_Error):

    """
    ML estimation of the random effects spatial error model with all results and
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
    sig2_u       : float
                   Sigma squared for random effects
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
                   Variance covariance matrix (k+2 x k+2), all coefficients
    vm1          : array
                   Variance covariance matrix (k+3 x k+3), includes sig2
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
    >>> import pandas as pd
    >>> import libpysal
    >>> import spreg
    >>> nat = libpysal.examples.load_example("NCOVR")
    >>> db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
    >>> nat_shp = libpysal.examples.get_path("NAT.shp")
    >>> w_full = libpysal.weights.Queen.from_shapefile(nat_shp)
    >>> name_y = ["HR70", "HR80", "HR90"]
    >>> y_full = np.array([db.by_col(name) for name in name_y]).T
    >>> name_x = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
    >>> x_full = np.array([db.by_col(name) for name in name_x]).T
    >>> name_c = ["STATE_NAME", "FIPSNO"]
    >>> df_counties = pd.DataFrame([db.by_col(name) for name in name_c], index=name_c).T
    >>> filter_states = ["Kansas", "Missouri", "Oklahoma", "Arkansas"]
    >>> filter_counties = df_counties[df_counties["STATE_NAME"].isin(filter_states)]["FIPSNO"].values
    >>> counties = np.array(db.by_col("FIPSNO"))
    >>> subid = np.where(np.isin(counties, filter_counties))[0]
    >>> w = w_subset(w_full, subid)
    >>> w.transform = 'r'
    >>> y = y_full[subid, ]
    >>> x = x_full[subid, ]
    >>> re_error = spreg.Panel_RE_Error(y, x, w, name_y=name_y, name_x=name_x, name_ds="NAT")
    Warning: Assuming panel is in wide format, i.e. y[:, 0] refers to T0, y[:, 1] refers to T1, etc.
    Similarly, assuming x[:, 0:T] refers to T periods of k1, x[:, T+1:2T] refers to k2, etc.
    >>> np.around(re_error.betas, decimals=4)
    array([[5.87893756],
           [3.23269025],
           [2.62996804],
           [0.34042682],
           [4.9782446]])
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
        bigy, bigx, name_y, name_x, warn = check_panel(y, x, w, name_y, name_x)
        set_warn(self, warn)
        bigx, name_x, warn = USER.check_constant(bigx, name_x)
        set_warn(self, warn)
        w = USER.check_weights(w, bigy, w_required=True, time=True)

        BasePanel_RE_Error.__init__(self, bigy, bigx, w, epsilon=epsilon)
        self.title = "MAXIMUM LIKELIHOOD SPATIAL ERROR PANEL" + " - RANDOM EFFECTS"
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_x, bigx, constant=False)
        self.name_x.append("lambda")
        self.name_x.append("sig2_u")  # error variance parameter
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
    sig2 = spdot(er.T, er) / (n * t)
    nlsig2 = (n * t / 2.0) * np.log(sig2)
    a = I - rho * Wsp
    LU = SuperLU(a.tocsc())
    jacob = t * np.sum(np.log(np.abs(LU.U.diagonal())))
    clike = nlsig2 - jacob
    return clike


def phi_c_loglik(phi, rho, beta, bigy, bigx, n, t, W_nt):
    # Demeaned variables
    y = demean_panel(bigy, n, t, phi=phi)
    x = demean_panel(bigx, n, t, phi=phi)
    # Lag dependent variable
    ylag = spdot(W_nt, y)
    er = y - rho * ylag - spdot(x, beta)
    sig2 = spdot(er.T, er)
    nlsig2 = (n * t / 2.0) * np.log(sig2)
    nphi2 = (n / 2.0) * np.log(phi ** 2)
    clike = nlsig2 - nphi2
    return clike


def err_c_loglik_ord(
    lam_phi, evals, evecs, n, t, bigy, bigx, ylag, xlag, y_mean, x_mean, I, Wsp
):
    # concentrated log-lik for error model, no constants, eigenvalues
    lam, phi = lam_phi
    cvals = t * phi ** 2 + 1 / (1 - lam * evals) ** 2
    P = spdot(np.diag(cvals ** (-0.5)), evecs.T)
    pr = P - (I - lam * Wsp)
    pr_nt = sp.kron(sp.identity(t), pr, format="csr")
    # Term 1
    yrand = bigy + spdot(pr_nt, y_mean)
    xrand = bigx + spdot(pr_nt, x_mean)
    ys = yrand - lam * ylag
    xs = xrand - lam * xlag
    ysys = np.dot(ys.T, ys)
    xsxs = np.dot(xs.T, xs)
    xsxsi = la.inv(xsxs)
    xsys = np.dot(xs.T, ys)
    x1 = np.dot(xsxsi, xsys)
    x2 = np.dot(xsys.T, x1)
    ee = ysys - x2
    sig2 = ee[0][0]
    nlsig2 = (n * t / 2.0) * np.log(sig2)
    # Term 2
    revals = t * phi ** 2 * (1 - lam * evals) ** 2
    phi_jacob = 1 / 2 * np.log(1 + revals).sum()
    # Term 3
    jacob = t * np.log(1 - lam * evals).sum()
    if isinstance(jacob, complex):
        jacob = jacob.real
    # this is the negative of the concentrated log lik for minimization
    clik = nlsig2 + phi_jacob - jacob
    return clik
