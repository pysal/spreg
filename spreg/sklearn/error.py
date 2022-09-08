__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

"""
Implement spatial error model in the format of scikit-learn
current goal is to make this work on its own, then progress down dependencies
"""

import numpy as np
from ..utils import optim_moments, get_spFilter
from ..sputils import spdot, sphstack
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from libpysal.weights import lag_spatial
from scipy.sparse.linalg import splu
from scipy.optimize import minimize_scalar


class Error(RegressorMixin, LinearModel):
    """
    Unified scikit-learn style estimator for spatial error models.

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
                   scalar corresponding to indirect effect on the spatial error term
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

    Fit spatial error model using default estimation method
    (generalized method of moments).

    >>> model = spreg.sklearn.Error(w=weights)
    >>> model = model.fit(X, y)
    >>> print(model.intercept_)
    [2.05623048]
    >>> print(model.coef_)
    [[ 0.02630646 -0.02201035]]
    >>> print(model.indir_coef_)
    -0.017493400401888144
    >>> print(model.score(X, y))
    0.5755042763160537

    Fit spatial error model using default estimation method (maximum
    likelihood).

    >>> model.fit(X, y, method="full")
    >>> print(model.intercept_)
    [1.79049179]
    >>> print(model.coef_)
    [[ 0.03330595 -0.02815453]]
    >>> print(model.indir_coef_)
    0.9913828713964514
    >>> print(model.score(X, y))
    0.5002107937316029
    """
    
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept

    def fit(self, X, y, yend=None, q=None, method="gm", epsilon=1e-7):
        """
        Fit spatial error model.
        
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
        self            : Error
                          fitted spreg.sklearn.Error object
        """
        
        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)
        y = y.reshape(-1, 1)  # ensure vector TODO FORMALIZE THIS

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        if self.fit_intercept:
            X = np.insert(X, 0, np.ones((X.shape[0],)), axis=1)

        method = method.lower()
        if method == "gm":
            self._fit_gm(X, y, yend, q)
        elif method in ["full", "lu", "ord"]:
            self._fit_ml(X, y, method, epsilon)
        else:
            raise ValueError(f"Method was {self.method}, choose 'gm', 'full'," +
                             " 'lu', or 'ord'")

        return self

    def _fit_gm(self, X, y, yend, q):
        """
        Helper method for fitting with GMM
        """

        def ols(X, y):
            xtx = spdot(X.T, X)
            xty = spdot(X.T, y)

            xtxi = np.linalg.inv(xtx)
            return np.dot(xtxi, xty)

        def tsls(X, y, yend, q):
            z = sphstack(X, yend)
            h = sphstack(X, q)
            hth = spdot(h.T, h)
            hthi = np.linalg.inv(hth)
            zth = spdot(z.T, h)
            hty = spdot(h.T, y)

            factor_1 = np.dot(zth, hthi)
            factor_2 = np.dot(factor_1, zth.T)
            varb = np.linalg.inv(factor_2)
            factor_3 = np.dot(varb, factor_1)
            return np.dot(factor_3, hty)

        # First stage
        if yend is not None:
            stage1 = tsls(X, y, yend, q)
        else:
            stage1 = ols(X, y)

        # Next, do generalized method of moments to calculate the error effect
        stage1_errors = (np.dot(X, stage1) - y).reshape(-1, 1)
        moments = self._moments_gm_error(self.w, stage1_errors)
        self.indir_coef_ = optim_moments(moments)

        # Generate estimated direct effects by filtering the variables
        xs = get_spFilter(self.w, self.indir_coef_, X)
        ys = get_spFilter(self.w, self.indir_coef_, y)

        if yend is not None:
            yend_s = get_spFilter(self.w, self.indir_coef_, yend)
            params_ = tsls(xs, ys, yend_s, q)
        else:
            params_ = ols(xs, ys)

        if self.fit_intercept:
            self.coef_ = params_[1:].T
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_.T

    def _fit_ml(self, X, y, method, epsilon):
        """
        Helper method for fitting with maximum likelihood
        """

        ylag = lag_spatial(self.w, y)
        xlag = lag_spatial(self.w, X)

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
                              args=(X, y, xlag, ylag, evals, method),
                              tol=epsilon)

        # Get coefficient estimates
        self.indir_coef_ = res.x
        ys = y - self.indir_coef_ * ylag
        xs = X - self.indir_coef_ * xlag
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        params_ = np.dot(xsxsi, xsys)

        if self.fit_intercept:
            self.coef_ = params_[1:].T
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_.T

    def _log_likelihood(self, lam, X, y, xlag, ylag, evals, method):
        """
        Defines log likelihoods for each of the three maximum likelihood methods
        """

        # Common stuff for all methods
        n = y.shape[0]
        ys = y - lam * ylag
        xs = X - lam * xlag
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        sig2 = ee[0][0] / n
        nlsig2 = (n / 2.0) * np.log(sig2)

        # Method-specific part
        if method == "full":
            a = -lam * self.w.full()[0]
            np.fill_diagonal(a, 1.0)
            jacob = np.log(np.linalg.det(a))
            # this is the negative of the concentrated log lik for minimization
        elif method == "lu":
            if isinstance(lam, np.ndarray):
                if lam.shape == (1, 1):
                    lam = lam[0][0]  # why does the interior value change?
            a = sp.identity(n) - lam * self.w.sparse
            LU = splu(a.tocsc())
            jacob = np.sum(np.log(np.abs(LU.U.diagonal())))
            # this is the negative of the concentrated log lik for minimization
        else:
            revals = lam * evals
            jacob = np.log(1 - revals).sum()
            if isinstance(jacob, complex):
                jacob = jacob.real
            # this is the negative of the concentrated log lik for minimization
        clik = nlsig2 - jacob
        return clik

    def _moments_gm_error(self, w, u):
        """
        Helper method to calculate moments for GMM estimation
        """

        try:
            wsparse = w.sparse
        except AttributeError:
            wsparse = w
        n = wsparse.shape[0]
        u2 = np.dot(u.T, u)
        wu = wsparse * u
        uwu = np.dot(u.T, wu)
        wu2 = np.dot(wu.T, wu)
        wwu = wsparse * wu
        uwwu = np.dot(u.T, wwu)
        wwu2 = np.dot(wwu.T, wwu)
        wuwwu = np.dot(wu.T, wwu)
        wtw = wsparse.T * wsparse
        trWtW = np.sum(wtw.diagonal())
        g = np.array([[u2[0][0], wu2[0][0], uwu[0][0]]]).T / n
        G = np.array(
            [[2 * uwu[0][0], -wu2[0][0], n], [2 * wuwwu[0][0], -wwu2[0][0], trWtW],
            [uwwu[0][0] + wu2[0][0], -wuwwu[0][0], 0.]]) / n
        return [G, g]
