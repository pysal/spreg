import numpy as np
from dataclasses import dataclass
#import .user_output as inputcheck
from abc import ABC, abstractmethod

"""
Guo et al. 2022
Data
- spatial domain u_i
- dataset (x_i, y_i
)

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


class SupervisedModel(ABC):
    def __init__(self, X, y=None):
        # accepts dataset and spatial domain

        #inputcheck.check_arrays(X, y)
        self.X = X
        self.y = y
        self.N, self.D = X.shape

    @abstractmethod
    def __call__(self, params):
        # Return value of the model on the inputted params
        pass

    @abstractmethod
    def _objective(self, params):
        # Objective for model
        pass

    @abstractmethod
    def fit(self):
        # Do optimization of model on objective and return self
        pass


@dataclass
class SupervisedModelResults:
    X: np.ndarray
    y: np.ndarray

    def __init__(self, model: SupervisedModel):
        pass

    def to_dict(self):
        return vars(self)
