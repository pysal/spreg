__author__ = "Tyler D. Hoffman pysal@tdhoffman.com"

"""
Implements Durbin Lag (lagged X and lagged y) and Durbin Error
(lagged X and spatial error) models. These are syntactic sugar for
alternative uses of the Lag and Error classes.
"""

import numpy as np
from .lag import Lag
from .error import Error
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.stats import pearsonr


class DurbinLag(Lag):
    """
    Syntactic sugar for fitting a spatial lag of X and spatial lag of Y model.
    
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

    >>> model = spreg.sklearn.DurbinLag(w=weights)
    >>> model = model.fit(X, y)
    >>> print(model.intercept_)
    [1.71836332]
    >>> print(model.coef_)
    [[ 0.05990476 -0.02792598 -0.01464177  0.00336733]]
    >>> print(model.indir_coef_)
    [0.06211914]
    >>> print(model.score(X, y))
    0.042140182774405066
    """

    def fit(self, X, y, yend=None, q=None, w_lags=1, lag_q=True, method="gm", epsilon=1e-7):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().fit(X, y, yend, q, w_lags, lag_q, method, epsilon)

    def score(self, X, y):
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().score(X, y)

    def _decision_function(self, X):
        check_is_fitted(self)

        return safe_sparse_dot(
            np.linalg.inv(np.eye(self.w.n) - self.indir_coef_ * self.w.full()[0]),
            safe_sparse_dot(X, self.coef_.T, dense_output=True), dense_output=True)

class DurbinError(Error):
    """
    Syntactic sugar for fitting a spatial lag of X and spatial error model.
    
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

    Fit spatial error model using default estimation method
    (generalized method of moments).

    >>> model = spreg.sklearn.DurbinError(w=weights)
    >>> model = model.fit(X, y)
    >>> print(model.intercept_)
    [2.43723301]
    >>> print(model.coef_)
    [[ 0.02694104 -0.00739781 -0.00053044 -0.00160177]]
    >>> print(model.indir_coef_)
    -0.046479712483854504
    >>> print(model.score(X, y))
    0.6541881277608026
    """

    def fit(self, X, y, yend=None, q=None, method="gm", epsilon=1e-7):
        """
        Fit Durbin Error model.
        """
        
        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().fit(X, y, yend, q, method, epsilon)

    def score(self, X, y, sample_weight=None):
        """
        Return R2 value for Durbin Error model.
        """

        X = self._validate_data(X, accept_sparse=True)

        if self.w is None:
            raise ValueError("w must be libpysal.weights.W object")

        X = np.hstack((X, self.w.full()[0] @ X))
        return super().score(X, y)

    def score(self, X, y):
        """
        Computes pseudo R2 for the spatial error model.
        """

        y_pred = self.predict(X)
        return float(pearsonr(y.flatten(), y_pred.flatten())[0]**2)

    def _decision_function(self, X):
        check_is_fitted(self)

        X = np.hstack((X, self.w.full()[0] @ X))
        return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
