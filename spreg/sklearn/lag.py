__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

"""
Implement spatial lag model in the format of scikit-learn
current goal is to make this work on its own, then progress down dependencies
"""

import numpy as np
from scipy import sparse as sp
from ..utils import set_endog
from ..w_utils import symmetrize
from ..sputils import spdot, sphstack, spfill_diagonal
from sklearn.base import RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model._base import LinearModel
from libpysal.weights import lag_spatial
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr


class Lag(RegressorMixin, LinearModel):
    """
    Unified scikit-learn style estimator for spatial lag models

    Parameters
    ----------
    w            : libpysal.weights.W
                   spatial weights object (always needed; default None)
    fit_intercept: boolean
                   when True, fits model with an intercept (default True)

    Attributes
    ----------
    w            : libpysal.weights.W
                   spatial weights object (always needed; default None)
    coef_        : array
                   kx1 array of estimated coefficients corresponding to direct effects
    indir_coef_  : float
                   scalar corresponding to indirect effect on the spatial lag term
    intercept_   : float
                   if fit_intercept is True, this is the intercept, otherwise undefined

    Examples
    --------

    >>> import spreg.sklearn
    >>> import numpy as np
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

    Fit spatial lag model using default estimation method
    (generalized method of moments).

    >>> model = spreg.sklearn.Lag(w=weights)
    >>> model = model.fit(X, y)
    >>> print(model.intercept_)
    [2.23457809]
    >>> print(model.coef_)
    [[ 0.02486769 -0.02033653]]
    >>> print(model.indir_coef_)
    [-0.00243977]
    >>> print(model.score(X, y))
    0.5773206900447707

    Fit spatial error model using default estimation method (maximum
    likelihood).

    >>> model.fit(X, y, method="full")
    >>> print(model.intercept_)
    [2.15295169]
    >>> print(model.coef_)
    [[ 0.02500276 -0.01998133]]
    >>> print(model.indir_coef_)
    -0.000928393589087051
    >>> print(model.score(X, y))
    0.5786266229308156
    """
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept

    def _decision_function(self, X):
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse=True, reset=False)
        return safe_sparse_dot(
            np.linalg.inv(np.eye(self.w.n) - self.indir_coef_ * self.w.full()[0]),
            safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True) + \
            self.intercept_

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True, method="gm", epsilon=1e-7):
        """
        Fit spatial lag model.

        Parameters
        ----------
        X               : array
                          nxk array of covariates
        y               : array
                          nx1 array of dependent variable
        yend            : array
                          nxp array of endogenous variables (default None)
        q               : array
                          nxp array of external endogenous variables to use as instruments
                          (should not contain any variables in X; default None)
        w_lags          : integer
                          orders of W to include as instruments for the spatially 
                          lagged dependent variable. For example, w_lags=1, then 
                          instruments are WX; if w_lags=2, then WX, WWX; and so on.
                          (default 1)
        method          : string
                          name of estimation method to use (default "gm"). available options
                          are: "gm" (generalized method of moments), "full" (brute force 
                          maximum likelihood), "lu" (LU sparse decomposition maximum 
                          likelihood), and "ord" (Ord eigenvalue method)
        epsilon         : float
                          tolerance to use for fitting maximum likelihood models
                          (default 1e-7)
        
        Returns
        -------
        self            : Lag 
                          fitted spreg.sklearn.Lag object
        """

        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)
        y = y.reshape(-1, 1)  # ensure vector TODO FORMALIZE THIS

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        if self.fit_intercept:
            X = np.insert(X, 0, np.ones((X.shape[0],)), axis=1)
        else:
            self.intercept_ = 0

        if method == "gm":
            self._fit_gm(X, y, yend, q, w_lags, lag_q)
        elif method in ["full", "lu", "ord"]:
            self._fit_ml(X, y, method, epsilon)
        else:
            raise ValueError(f"Method was {self.method}, choose 'gm', 'full'," +
                             "'lu', or 'ord'")

        return self

    def _fit_gm(self, X, y, yend, q, w_lags, lag_q):
        """
        Helper method for fitting with GMM
        """

        yend2, q2 = set_endog(y, X[:, 1:], self.w,
                              yend, q, w_lags, lag_q)  # assumes constant in first column

        # including exogenous and endogenous variables
        z = sphstack(X, yend2)
        h = sphstack(X, q2)
        # k = number of exogenous variables and endogenous variables
        hth = spdot(h.T, h)
        hthi = np.linalg.inv(hth)
        zth = spdot(z.T, h)
        hty = spdot(h.T, y)

        factor_1 = np.dot(zth, hthi)
        factor_2 = np.dot(factor_1, zth.T)
        # this one needs to be in cache to be used in AK
        varb = np.linalg.inv(factor_2)
        factor_3 = np.dot(varb, factor_1)
        params_ = np.dot(factor_3, hty)

        if self.fit_intercept:
            self.coef_ = params_[1:-1].T
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_[:-1].T
        self.indir_coef_ = params_[-1]
    
    def _fit_ml(self, X, y, method, epsilon):
        """
        Helper method for fitting with maximum likelihood
        """

        ylag = lag_spatial(self.w, y)
        xtx = spdot(X.T, X)
        xtxi = np.linalg.inv(xtx)
        xty = spdot(X.T, y)
        xtyl = spdot(X.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = y - spdot(X, b0)
        e1 = ylag - spdot(X, b1)

        print("ml")
        # Create eigenvalues before minimizing
        if method == "ord" and self.w.asymmetry(intrinsic=False) == []:
            ww = symmetrize(self.w)
            WW = np.array(ww.todense())
            evals = np.linalg.eigvalsh(WW)
        elif method != "ord":
            evals = None
        else:
            evals = np.linalg.eigvals(self.w.full()[0])

        res = minimize_scalar(self._log_likelihood, bounds=(-1.0, 1.0),
                              args=(e0, e1, evals, method), method="bounded",
                              tol=epsilon)

        self.indir_coef_ = res.x[0][0]
        params_ = b0 - self.indir_coef_ * b1

        if self.fit_intercept:
            self.coef_ = params_[1:].T
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_.T

    def _log_likelihood(self, rho, e0, e1, evals, method):
        """
        Defines log likelihoods for each of the three maximum likelihood methods
        """

        n = self.w.n
        er = e0 - rho * e1
        sig2 = spdot(er.T, er) / n
        nlsig2 = (n / 2.0) * np.log(sig2)

        if method == "full":
            a = -rho * self.w.full()[0]
            spfill_diagonal(a, 1.0)
            jacob = np.log(np.linalg.det(a))
        elif method == "lu":
            if isinstance(rho, np.ndarray) and rho.shape == (1, 1):
                rho = rho[0][0]
            a = sp.identity(self.w.n)
            LU = sp.linalg.splu(a.tocsc())
            jacob = np.sum(np.log(np.abs(LU.U.diagonal())))
        else:
            revals = rho * evals
            jacob = np.log(1 - revals).sum()
            if isinstance(jacob, complex):
                jacob = jacob.real

        clik = nlsig2 - jacob
        return clik

    def score(self, X, y):
        """
        Computes pseudo R2 for the spatial lag model.
        """

        y_pred = self.predict(X)
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)
