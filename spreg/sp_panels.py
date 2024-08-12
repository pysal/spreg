"""
Spatial random effects panel model based on: :cite:`KKP2007`
"""

__author__ = (
    "Luc Anselin anselin@uchicago.edu, Pedro Amaral pedroamaral@cedeplar.ufmg.br"
)

from scipy import sparse as SP
import numpy as np
from . import ols as OLS
from .utils import optim_moments, RegressionPropsY, get_spFilter, spdot, set_warn
from . import user_output as USER
from . import summary_output as SUMMARY
from . import regimes as REGI

# import warnings


__all__ = ["GM_KKP"]


class BaseGM_KKP(RegressionPropsY):

    '''
    Base GMM method for a spatial random effects panel model based on
    Kapoor, Kelejian and Prucha (2007) :cite:`KKP2007`.

    Parameters
    ----------
    y          : array
                 n*tx1 array for dependent variable
    x          : array
                 Two dimensional array with n*t rows and one column for each
                 independent (exogenous) variable
                 (note: must already include constant term)
    w          : spatial weights object
                 Spatial weights matrix
    full_weights: boolean
                  Considers different weights for each of the 6 moment
                  conditions if True or only 2 sets of weights for the
                  first 3 and the last 3 monent conditions if False (default)

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    """
    '''

    def __init__(self, y, x, w, full_weights=False):

        # 1a. OLS --> \tilde{\delta}
        ols = OLS.BaseOLS(y=y, x=x)
        self.x, self.y, self.n, self.k, self.xtx = ols.x, ols.y, ols.n, ols.k, ols.xtx
        N = w.n
        T = y.shape[0] // N
        moments, trace_w2 = _moments_kkp(w.sparse, ols.u, 0)
        lambda1, sig_v = optim_moments(moments, all_par=True)
        Tw = SP.kron(SP.identity(T), w.sparse)
        ub = Tw.dot(ols.u)
        ulu = ols.u - lambda1 * ub
        Q1 = SP.kron(np.ones((T, T)) / T, SP.identity(N))
        sig_1 = float(np.dot(ulu.T, Q1.dot(ulu)) / N)
        # print('initial_lamb_sig:',lambda1,sig_v,sig_1)
        # print('theta:', 1 - np.sqrt(sig_v)/ np.sqrt(sig_1))
        Xi_a = SP.diags([(sig_v * sig_v) / (T - 1), sig_1 * sig_1])
        if full_weights:
            Tau = _get_Tau(w.sparse, trace_w2)
        else:
            Tau = SP.identity(3)
        Xi = SP.kron(Xi_a, Tau)
        moments_b, _ = _moments_kkp(w.sparse, ols.u, 1, trace_w2)
        G = np.vstack((np.hstack((moments[0], np.zeros((3, 1)))), moments_b[0]))
        moments6 = [G, np.vstack((moments[1], moments_b[1]))]
        lambda2, sig_vb, sig_1b = optim_moments(
            moments6, vcX=Xi.toarray(), all_par=True, start=[lambda1, sig_v, sig_1]
        )
        # 2a. reg -->\hat{betas}
        theta = 1 - np.sqrt(sig_vb) / np.sqrt(sig_1b)
        # print('theta:', theta)
        gls_w = SP.identity(N * T) - theta * Q1

        # With omega
        xs = gls_w.dot(get_spFilter(w, lambda2, x))
        ys = gls_w.dot(get_spFilter(w, lambda2, y))
        ols_s = OLS.BaseOLS(y=ys, x=xs)
        self.predy = spdot(self.x, ols_s.betas)
        self.u = self.y - self.predy
        self.vm = ols_s.vm  # Check
        self.betas = np.vstack((ols_s.betas, lambda2, sig_vb, sig_1b))
        self.e_filtered = self.u - lambda2 * SP.kron(SP.identity(T), w.sparse).dot(
            self.u
        )
        self.t, self.n = T, N
        self._cache = {}


class GM_KKP(BaseGM_KKP, REGI.Regimes_Frame):

    '''
    GMM method for a spatial random effects panel model based on
    Kapoor, Kelejian and Prucha (2007) :cite:`KKP2007`.

    Parameters
    ----------
    y          : array
                 n*tx1 or nxt array for dependent variable
    x          : array
                 Two dimensional array with n*t rows and k columns for
                 independent (exogenous) variable or n rows and k*t columns
                 (note, must not include a constant term)
    w          : spatial weights object
                 Spatial weights matrix, nxn
    full_weights: boolean
                  Considers different weights for each of the 6 moment
                  conditions if True or only 2 sets of weights for the
                  first 3 and the last 3 moment conditions if False (default)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'y'.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string or list of strings
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output

    Attributes
    ----------
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    t            : integer
                   Number of time periods
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    vm           : array
                   Variance covariance matrix (kxk)
    chow         : tuple
                   Contains 2 elements. 1: Pair of Wald statistic and p-value
                   for the setup of global regime stability. 2: array with Wald
                   statistic (col 0) and its p-value (col 1) for each beta that
                   varies across regimes.
                   Exists only if regimes is not None.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    title        : string
                   Name of the regression method used
    """
    Examples
    --------
    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.
    >>> from spreg import GM_KKP
    >>> import numpy as np
    >>> import libpysal
    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile. Note that
    libpysal.io.open() also reads data in CSV format; The GM_KKP function requires
    data to be passed in as numpy arrays, hence the user can read their
    data in using any method.
    >>> nat = libpysal.examples.load_example('NCOVR')
    >>> db = libpysal.io.open(nat.get_path("NAT.dbf"),'r')
    Extract the HR (homicide rates) data in the 70's, 80's and 90's from the DBF file
    and make it the dependent variable for the regression. Note that the data can also
    be passed in the long format instead of wide format (i.e. a vector with n*t rows
    and a single column for the dependent variable and a matrix of dimension n*txk
    for the independent variables).
    >>> name_y = ['HR70','HR80','HR90']
    >>> y = np.array([db.by_col(name) for name in name_y]).T
    Extract RD and PS in the same time periods from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxk*t numpy array, where k is the number of independent variables (not
    including a constant) and t is the number of time periods. Data must be
    organized in a way that all time periods of a given variable are side-by-side
    and in the correct time order.
    By default a vector of ones will be added to the independent variables passed in.
    >>> name_x = ['RD70','RD80','RD90','PS70','PS80','PS90']
    >>> x = np.array([db.by_col(name) for name in name_x]).T
    Since we want to run a spatial error panel model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``NAT.shp``.
    >>> w = libpysal.weights.Queen.from_shapefile(libpysal.examples.get_path("NAT.shp"))
    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, his allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:
    >>> w.transform = 'r'
    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional. In this example
    we set full_weights to False (the default), indicating that we will use
    only 2 sets of moments weights for the first 3 and the last 3 moment conditions.
    >>> reg = GM_KKP(y,x,w,full_weights=False,name_y=name_y, name_x=name_x)
    Warning: Assuming time data is in wide format, i.e. y[0] refers to T0, y[1], refers to T1, etc.
     Similarly, assuming x[0:k] refers to independent variables for T0, x[k+1:2k] refers to T1, etc.
    Once we have run the model, we can explore a little bit the output. We can
    either request a printout of the results with the command print(reg.summary) or
    check out the individual attributes of GM_KKP:
    >>> print(reg.summary)
    REGRESSION
    ----------
    SUMMARY OF OUTPUT: GM SPATIAL ERROR PANEL MODEL - RANDOM EFFECTS (KKP)
    ----------------------------------------------------------------------
    Data set            :     unknown
    Weights matrix      :     unknown
    Dependent Variable  :          HR                Number of Observations:        3085
    Mean dependent var  :      6.4983                Number of Variables   :           3
    S.D. dependent var  :      6.9529                Degrees of Freedom    :        3082
    Pseudo R-squared    :      0.3248
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     z-Statistic     Probability
    ------------------------------------------------------------------------------------
                CONSTANT       6.4922156       0.1126713      57.6208690       0.0000000
                      RD       3.6244575       0.0877475      41.3055536       0.0000000
                      PS       1.3118778       0.0852516      15.3883058       0.0000000
                  lambda       0.4177759
                sigma2_v      22.8190822
                sigma2_1      39.9099323
    ------------------------------------------------------------------------------------
    ================================ END OF REPORT =====================================
    >>> print(reg.name_x)
    ['CONSTANT', 'RD', 'PS', 'lambda', ' sigma2_v', 'sigma2_1']
    The attribute reg.betas contains all the coefficients: betas, the spatial error
    coefficient lambda, sig2_v and sig2_1:
    >>> print(np.around(reg.betas,4))
    [[ 6.4922]
     [ 3.6245]
     [ 1.3119]
     [ 0.4178]
     [22.8191]
     [39.9099]]
    Finally, we can check the standard erros of the betas:
    >>> print(np.around(np.sqrt(reg.vm.diagonal().reshape(3,1)),4))
    [[0.1127]
     [0.0877]
     [0.0853]]
    '''

    def __init__(
        self,
        y,
        x,
        w,
        full_weights=False,
        regimes=None,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
    ):
        n_rows = USER.check_arrays(y, x)
        bigy, bigx, name_y, name_x = _get_panel_data(y, x, w, name_y, name_x)
        w = USER.check_weights(w, bigy, w_required=True, time=True)
        x_constant, name_x, warn = USER.check_constant(bigx, name_x)
        set_warn(self, warn)
        self.title = "GM SPATIAL ERROR PANEL MODEL - RANDOM EFFECTS (KKP)"
        self.name_x = USER.set_name_x(name_x, x_constant)

        if regimes is not None:
            self.regimes = regimes
            self.name_regimes = USER.set_name_ds(name_regimes)
            regimes_l = self._set_regimes(w, bigy.shape[0])
            self.name_x_r = self.name_x
            x_constant, self.name_x = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes_l,
                constant_regi=False,
                cols2regi="all",
                names=self.name_x,
            )

        BaseGM_KKP.__init__(self, bigy, x_constant, w, full_weights=full_weights)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x.extend(["lambda", " sigma2_v", "sigma2_1"])
        self.name_w = USER.set_name_w(name_w, w)
        if regimes is not None:
            self.kf += 3
            self.chow = REGI.Chow(self)
            self.title += " WITH REGIMES"
            regimes = True
        SUMMARY.GM_Panels(reg=self, w=w, vm=vm, regimes=regimes)

    def _set_regimes(self, w, n_rows):  # Must add case for regime_err_sep = True
        self.constant_regi = "many"
        self.cols2regi = "all"
        self.regime_err_sep = False
        self.regimes_set = REGI._get_regimes_set(self.regimes)
        if len(self.regimes) == w.n:
            regimes_l = self.regimes * (n_rows // w.n)
        elif len(self.regimes) == n_rows:
            regimes_l = self.regimes
        else:
            raise Exception("The lenght of 'regimes' must be either equal to n or n*t.")
        return regimes_l


def _moments_kkp(ws, u, i, trace_w2=None):
    """
    Compute G and g matrices for the KKP model.
    ...

    Parameters
    ----------
    ws          : Sparse matrix
                  Spatial weights sparse matrix
    u           : array
                  Residuals. nx1 array assumed to be aligned with w

    i       : integer
                  0 if Q0, 1 if Q1
    trace_w2    : float
                  trace of WW. Computed in 1st step and saved for step 2.
    Returns
    -------
    moments     : list
                  List of two arrays corresponding to the matrices 'G' and
                  'g', respectively.
    trace_w2    : float
                  trace of WW. Computed in 1st step and saved for step 2.
    """
    N = ws.shape[0]
    T = u.shape[0] // N
    if i == 0:
        Q = SP.kron(SP.identity(T) - np.ones((T, T)) / T, SP.identity(N))
    else:
        Q = SP.kron(np.ones((T, T)) / T, SP.identity(N))
    Tw = SP.kron(SP.identity(T), ws)
    ub = Tw.dot(u)
    ubb = Tw.dot(ub)
    Qu = Q.dot(u)
    Qub = Q.dot(ub)
    Qubb = Q.dot(ubb)
    G11 = float(2 * np.dot(u.T, Qub))
    G12 = float(-np.dot(ub.T, Qub))
    G21 = float(2 * np.dot(ubb.T, Qub))
    G22 = float(-np.dot(ubb.T, Qubb))
    G31 = float(np.dot(u.T, Qubb) + np.dot(ub.T, Qub))
    G32 = float(-np.dot(ub.T, Qubb))
    if trace_w2 == None:
        trace_w2 = (ws.power(2)).sum()
    G23 = ((T - 1) ** (1 - i)) * trace_w2
    if i == 0:
        G = np.array(
            [[G11, G12, N * (T - 1) ** (1 - i)], [G21, G22, G23], [G31, G32, 0]]
        ) / (N * (T - 1) ** (1 - i))
    else:
        G = np.array(
            [
                [G11, G12, 0, N * (T - 1) ** (1 - i)],
                [G21, G22, 0, G23],
                [G31, G32, 0, 0],
            ]
        ) / (N * (T - 1) ** (1 - i))
    g1 = float(np.dot(u.T, Qu))
    g2 = float(np.dot(ub.T, Qub))
    g3 = float(np.dot(u.T, Qub))
    g = np.array([[g1, g2, g3]]).T / (N * (T - 1) ** (1 - i))
    return [G, g], trace_w2


def _get_Tau(ws, trace_w2):
    """
    Computes Tau as in :cite:`KKP2007`.
    ...

    Parameters
    ----------
    ws          : Sparse matrix
                  Spatial weights sparse matrix
    trace_w2    : float
                  trace of WW. Computed in 1st step of _moments_kkp
    """
    N = ws.shape[0]
    T12 = 2 * trace_w2 / N
    wtw = ws.T.dot(ws)
    T22 = wtw.power(2).sum()
    wtpw = ws.T + ws
    T23 = wtw.multiply(wtpw).sum()
    d_wwpwtw = ws.multiply(ws.T).sum(0) + wtw.diagonal()
    T33 = d_wwpwtw.sum()
    Tau = np.array([[2 * N, T12, 0], [T12, T22, T23], [0, T23, T33]]) / N
    return Tau


def _get_panel_data(y, x, w, name_y, name_x):
    """
    Performs some checks on the data structure and converts from wide to long if needed.
    ...

    Parameters
    ----------
    y          : array
                 n*tx1 or nxt array for dependent variable
    x          : array
                 Two dimensional array with n*t rows and k columns for
                 independent (exogenous) variable or n rows and k*t columns
                 (note, must not include a constant term)
    name_y       : string or list of strings
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    """

    if y.shape[0] / w.n != y.shape[0] // w.n:
        raise Exception("y must be ntx1 or nxt, and w must be an nxn PySAL W object.")
    N, T = y.shape[0], y.shape[1]
    k = x.shape[1] // T
    if x.shape[0] != N and x.shape[0] != N * T:
        raise Exception(
            "X must have either n rows and k*t columns or n*t rows and k columns."
        )
    if x.shape[1] != k and x.shape[1] != k * T:
        raise Exception(
            "X must have either n rows and k*t columns or n*t rows and k columns."
        )
    if y.shape[1] > 1:
        message = (
            "Assuming time data is in wide format, i.e. y[0] refers to T0, y[1], refers to T1, etc."
            "\n Similarly, assuming x[0:k] refers to independent variables for T0, x[k+1:2k] refers to T1, etc."
        )
        print("Warning: " + message)
        # warnings.warn(message)

        if y.shape[1] != T:
            raise Exception(
                "y in wide format must have t columns and be compatible with x's k*t columns."
            )

        bigy = y.reshape((y.size, 1), order="F")

        bigx = x[:, 0:T].reshape((N * T, 1), order="F")
        for i in range(1, k):
            bigx = np.hstack(
                (bigx, x[:, T * i : T * (i + 1)].reshape((N * T, 1), order="F"))
            )
    else:
        bigy, bigx = y, x

    if name_y:
        if not isinstance(name_y, str) and not isinstance(name_y, list):
            raise Exception("name_y must either be strings or a list of strings.")
        if len(name_y) > 1 and isinstance(name_y, list):
            name_y = "".join([i for i in name_y[0] if not i.isdigit()])
        if len(name_y) == 1 and isinstance(name_y, list):
            name_y = name_y[0]
    if name_x:
        if len(name_x) != k * T and len(name_x) != k:
            raise Exception(
                "Names of columns in X must have exactly either k or k*t elements."
            )
        if len(name_x) > k:
            name_bigx = []
            for i in range(k):
                name_bigx.append("".join([j for j in name_x[i * T] if not j.isdigit()]))
            name_x = name_bigx

    return bigy, bigx, name_y, name_x


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()
