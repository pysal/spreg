"""
Proposed Lag model interface
"""

import numpy as np
from .utils import set_endog
from .sputils import spdot, sphstack
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel


class Lag(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True, method="gm"):
        self.w = w
        self.fit_intercept = fit_intercept
        self.method = method

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True):
        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)
        y = y.reshape(-1, 1)  # ensure vector TODO FORMALIZE THIS

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        if self.fit_intercept:
            X = np.insert(X, 0, np.ones((X.shape[0],)), axis=1)

        if self.method == "gm":
            self._fit_gm(X, y, yend, q, w_lags, lag_q)
        elif self.method in ["full", "lu", "ord"]:
            self._fit_ml(X, y)
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
        params = np.dot(factor_3, hty)

        self.intercept_ = params[0]
        self.coef_ = params[1:-1]
        self.indir_coef_ = params[-1]
    
    def _fit_ml(self, X, y):
        pass

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

    #fields = ["RMSQ", "AGE", "LOGDIS", "LOGRAD", "TAX", "PTRATIO",
              #"TRANSB", "LOGSTAT", "CRIM", "ZN", "INDUS", "CHAS", "NOXSQ"]
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
