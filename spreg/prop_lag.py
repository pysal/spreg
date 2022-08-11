"""
Proposed Lag model interface
"""

import numpy as np
from .utils import spdot
from .supervised_model import SupervisedModel, SupervisedModelResults
from libpysal.weights.spatial_lag import lag_spatial


class Lag(SupervisedModel):
    """
    Order of params: intercept, direct, indirect
    """
    def __init__(self, X, y, w):
        super().__init_(X, y)
        self.w = w

    def __call__(self, params):
        return spdot(self.X, params[1:-1]) + \
               params[-1] * lag_spatial(self.w, self.y) + \
               params[0]

    def _objective(self, params):
        pass

    def fit(self, method="gm", yend=None, q=None, epsilon=1e-7):
        method = method.lower()
        if method in ["full", "lu", "ord"]:
            self._fit_ml(method, epsilon=1e-7)
        elif method == "gm":
            self._fit_gm(yend, q)
        else:
            raise ValueError(f"{method} is unsupported, choose " + \
                             "'full', 'lu', 'ord', or 'gm'")
        return self  # maybe return SupervisedModelResults

    def _fit_gm(self, yend, q):
        pass
    
    def _fit_ml(self, method, epsilon):
        pass

if __name__ == "__main__":
    pass
