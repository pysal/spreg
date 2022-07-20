import numpy as np

"""
Guo et al. 2022
Data
- spatial domain u_i
- dataset (x_i, y_i)

Model
- model as function of params M(theta)
- fitting algorithm f

Objective
- loss function L

Output
- theta hat
- more?

Issues:
* spatial domain is hard to encode in general (duh)
  - perhaps leave that out of the general class and specialize based on kind of model
- oftentimes the fit method isn't actually an optimization procedure
  e.g OLS where the L2 optimum is known and has a direct solution
  --> does it make sense to add a full fit method with optimization
      as well as a direct solution? what about heuristic solutions?
- add immutable fields/properties
- results can be a python3 data class with a nice repr so .summary returns an easy display

spreg has...
- spatial regression models
- regimes models
- seemingly-unrelated regression models
- diagnostics
"""


class GenericModel:
    def __init__(self, X, y=None):
        # accepts dataset and spatial domain
        self.X = X
        self.y = y
        self.N, self.D = X.shape

    def _model(self, params):
        # Return value of the model on the inputted params
        pass

    def _objective(self):
        # Return objective for fitting model to
        pass

    def fit(self):
        # Do optimization of model on objective
        # return self
        pass

    def summary(self):
        # Return some kind of set of results
        # might return a results container class
        pass


class OLS(GenericModel):
    def __init__(self, X, y):
        super.__init__()

    def _model(self, params):
        return self.X @ params[1:] + params[0]

    def _objective(self, params):
        return np.sum((self._model(params) - self.y)**2)

    def fit(self):
        # optimize objective
        pass

    def summary(self):
        # return results
        pass


class SLX(GenericModel):
    def __init__(self, X, y):
        super.__init__()

    def _model(self, params):
        return params[0] + self.X @ params[1:self.D + 1] + \
            self.W @ self.X @ params[self.D + 1:]

    def _objective(self, params):
        return np.sum((self._model(params) - self.y)**2)

    def fit(self):
        # optimize objective
        pass

    def summary(self):
        # return results
        pass
