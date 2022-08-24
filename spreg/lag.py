__author__ = "Tyler D. Hoffman tdhoffman@asu.edu"

"""
Implement spatial lag model in the format of scikit-learn
current goal is to make this work on its own, then progress down dependencies
"""

import numpy as np
from scipy import sparse as sp
from .utils import set_endog
from .sputils import spdot, sphstack, spfill_diagonal
from sklearn.base import RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model._base import LinearModel
from libpysal.weights import lag_spatial
from scipy.optimize import minimize_scalar


class Lag(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True):
        self.w = w
        self.fit_intercept = fit_intercept

    def _decision_function(self, X, y):
        check_is_fitted(self)

        X, y = self._validate_data(X, y, accept_sparse=True)
        return safe_sparse_dot(
            np.linalg.inv(sp.eye(y.shape[0]) - self.indir_coef_ * self.w),
            safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True)

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True, method="gm", epsilon=1e-7):
        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)
        y = y.reshape(-1, 1)  # ensure vector TODO FORMALIZE THIS

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        if self.fit_intercept:
            X = np.insert(X, 0, np.ones((X.shape[0],)), axis=1)

        if method == "gm":
            self._fit_gm(X, y, yend, q, w_lags, lag_q)
        elif method in ["full", "lu", "ord"]:
            self._fit_ml(X, y, method, epsilon)
        else:
            raise ValueError(f"Method was {self.method}, choose 'gm', 'full'," +
                             "'lu', or 'ord'")

        return self

    def _fit_gm(self, X, y, yend, q, w_lags, lag_q):
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
            self.coef_ = params_[1:-1]
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_[:-1]
        self.indir_coef_ = params_[-1]
    
    def _fit_ml(self, X, y, method, epsilon):
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
            self.coef_ = params_[1:]
            self.intercept_ = params_[0]
        else:
            self.coef_ = params_

    def _log_likelihood(self, rho, e0, e1, evals, method):
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


if __name__ == "__main__":
    import spreg
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from libpysal.examples import load_example
    from libpysal.weights import Kernel, fill_diagonal

    boston = load_example("Bostonhsg")
    boston_df = gpd.read_file(boston.get_path("boston.shp"))

    boston_df["NOXSQ"] = (10 * boston_df["NOX"])**2
    boston_df["RMSQ"] = boston_df["RM"]**2
    boston_df["LOGDIS"] = np.log(boston_df["DIS"].values)
    boston_df["LOGRAD"] = np.log(boston_df["RAD"].values)
    boston_df["TRANSB"] = boston_df["B"].values / 1000
    boston_df["LOGSTAT"] = np.log(boston_df["LSTAT"].values)

    fields = ["RMSQ", "CRIM"]
    X = boston_df[fields].values
    y = np.log(boston_df["CMEDV"].values)  # predict log corrected median house prices from covars

    weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    weights = fill_diagonal(weights, 0)

    model = spreg.Lag(w=weights)
    model.fit(X, y)
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)

    old_model = spreg.GM_Lag(y, X, w=weights)
    print(old_model.betas)

    model.fit(X, y, method="full")
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)

    old_model = spreg.ML_Lag(y, X, w=weights, method="full")
    print(old_model.betas)

    model.fit(X, y, method="lu")
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)

    old_model = spreg.ML_Lag(y, X, w=weights, method="lu")
    print(old_model.betas)

    model.fit(X, y, method="ord")
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)

    old_model = spreg.ML_Lag(y, X, w=weights, method="ord")
    print(old_model.betas)
