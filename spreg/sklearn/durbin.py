__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

"""
Implements Durbin Lag (lagged X and lagged y) and Durbin Error
(lagged X and spatial error) models. These are syntactic sugar for
alternative uses of the Lag and Error classes.
"""

import numpy as np
from .lag import Lag
from .error import Error


class DurbinLag(Lag):
    """
    Should inherit everything properly from Lag.
    """

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True, method="gm", epsilon=1e-7):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().fit(X, y, yend, q, w_lags, lag_q, method, epsilon)

    def score(self, X, y, sample_weight=None):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().score(X, y, sample_weight=sample_weight)


class DurbinError(Error):
    """
    Should inherit everything properly from Error.
    """

    def fit(self, X, y, yend=None, q=None, method="gm", epsilon=1e-7):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().fit(X, y, yend, q, method, epsilon)

    def score(self, X, y, sample_weight=None):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().score(X, y, sample_weight=sample_weight)


if __name__ == "__main__":
    import spreg
    import numpy as np
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
    
    model = spreg.DurbinLag(w=weights)
    model.fit(X, y)
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)

    model = spreg.DurbinError(w=weights)
    model.fit(X, y)
    print(model.intercept_)
    print(model.coef_)
    print(model.indir_coef_)
