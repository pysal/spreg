import numpy as np
from .utils import optim_moments, get_spFilter
from .sputils import spdot
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model._base import LinearModel

class Error(RegressorMixin, LinearModel):
    def __init__(self, w=None, fit_intercept=True, method="gm"):
        self.w = w
        self.fit_intercept = fit_intercept
        self.method = method

    def fit(self, X, y):
        # Input validation
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True)

        if self.method == "gm":
            return self._fit_gm(X, y)
        elif self.method in ["full", "lu", "ord"]:
            return self._fit_ml(X, y)
        else:
            raise ValueError(f"Method was {self.method}, choose 'gm', 'full'," +
                             " 'lu', or 'ord'")

    def _fit_gm(self, X, y):
        def ols(X, y):
            xtx = spdot(X.T, X)
            xty = spdot(X.T, y)

            xtxi = np.linalg.inv(xtx)
            return np.dot(xtxi, xty)

        # First, fit an OLS
        ols1 = ols(X, y)
        
        # Next, do generalized method of moments to calculate the error effect
        print(np.dot(X, ols1.T) - y)
        moments = _moments_gm_error(self.w, np.dot(X, ols1.T) - y)
        self.indir_coefs_ = optim_moments(moments)

        # Generate estimated direct effects by filtering the variables
        xs = get_spFilter(w, lambda1, X)
        ys = get_spFilter(w, lambda1, y)
        self.dir_coefs_ = ols(xs, ys)

        return self


def _moments_gm_error(w, u):
    try:
        wsparse = w.sparse
    except:
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

    fields = ["RMSQ", "AGE", "LOGDIS", "LOGRAD", "TAX", "PTRATIO",
              "TRANSB", "LOGSTAT", "CRIM", "ZN", "INDUS", "CHAS", "NOXSQ"]
    X = boston_df[fields].values
    y = np.log(boston_df["CMEDV"].values)  # predict log corrected median house prices from covars

    weights = Kernel(boston_df[["x", "y"]], k=50, fixed=False)
    weights = fill_diagonal(weights, 0)

    model = spreg.Error(w=weights)
    model.fit(X, y)
