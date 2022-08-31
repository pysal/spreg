"""
Spatial Error SUR estimation
"""

__author__ = "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"


import numpy as np
import numpy.linalg as la
from scipy import stats

stats.chisqprob = stats.chi2.sf
from . import summary_output as SUMMARY
from . import user_output as USER
from . import regimes as REGI
from scipy.sparse.linalg import splu as SuperLU
from scipy.optimize import minimize_scalar, minimize
from scipy import sparse as sp

from .ml_error import err_c_loglik_sp
from .utils import optim_moments
from .sur_utils import (
    sur_dictxy,
    sur_corr,
    sur_dict2mat,
    sur_crossprod,
    sur_est,
    sur_resids,
    filter_dict,
    check_k,
)
from .sur import BaseSUR, _sur_ols
from .diagnostics_sur import sur_setp, lam_setp, sur_chow
from .regimes import buildR, wald_test

__all__ = ["BaseSURerrorGM", "SURerrorGM", "BaseSURerrorML", "SURerrorML"]


class BaseSURerrorGM:
    """Base class for SUR Error estimation by Generalized Moment Estimation

    Parameters
    ----------
    bigy       : dictionary
                 with vector for dependent variable by equation
    bigX       : dictionary
                 with matrix of explanatory variables by equation
                 (note, already includes constant term)
    w          : spatial weights object

    Attributes
    ----------
    n_eq       : int
                 number of equations
    n          : int
                 number of observations in each cross-section
    bigy       : dictionary
                 with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary
                 with matrices of explanatory variables,
                 one for each equation
    bigK       : array
                 n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : dictionary
                 spatially lagged dependent variable
    bigXlag    : dictionary
                 spatially lagged explanatory variable
    lamsur     : float
                 spatial autoregressive coefficient in GM SUR Error
    bSUR       : array
                 beta coefficients in GM SUR Error
    varb       : array
                 variance of beta coefficients in GM SUR Error
    sig        : array
                 error variance-covariance matrix in GM SUR Error
    corr       : array
                 error correlation matrix
    bigE       : array
                 n by n_eq matrix of vectors of residuals for each equation

    """

    def __init__(self, bigy, bigX, w):
        self.n = w.n
        self.n_eq = len(bigy.keys())
        WS = w.sparse
        I = sp.identity(self.n)
        # variables
        self.bigy = bigy
        self.bigX = bigX
        # number of variables by equation
        self.bigK = np.zeros((self.n_eq, 1), dtype=np.int_)
        for r in range(self.n_eq):
            self.bigK[r] = self.bigX[r].shape[1]

        # OLS
        self.bigXX, self.bigXy = sur_crossprod(self.bigX, self.bigy)
        reg0 = _sur_ols(self)

        # Moments
        moments = _momentsGM_sur_Error(WS, reg0.olsE)
        lam = np.zeros((self.n_eq, 1))
        for r in range(self.n_eq):
            lam[r] = optim_moments(moments[r])

        # spatially lagged variables
        self.bigylag = {}
        for r in range(self.n_eq):
            self.bigylag[r] = WS * self.bigy[r]
        # note: unlike WX as instruments, this includes the constant
        self.bigXlag = {}
        for r in range(self.n_eq):
            self.bigXlag[r] = WS * self.bigX[r]

        # spatially filtered variables
        sply = filter_dict(lam, self.bigy, self.bigylag)
        splX = filter_dict(lam, self.bigX, self.bigXlag)
        WbigE = WS * reg0.olsE
        splbigE = reg0.olsE - WbigE * lam.T
        splXX, splXy = sur_crossprod(splX, sply)
        b1, varb1, sig1 = sur_est(splXX, splXy, splbigE, self.bigK)
        bigE = sur_resids(self.bigy, self.bigX, b1)

        self.lamsur = lam
        self.bSUR = b1
        self.varb = varb1
        self.sig = sig1
        self.corr = sur_corr(self.sig)
        self.bigE = bigE


def _momentsGM_sur_Error(w, u):
    n = w.shape[0]
    u2 = (u * u).sum(0)
    wu = w * u
    uwu = (u * wu).sum(0)
    wu2 = (wu * wu).sum(0)
    wwu = w * wu
    uwwu = (u * wwu).sum(0)
    wwu2 = (wwu * wwu).sum(0)
    wwuwu = (wwu * wu).sum(0)
    trWtW = w.multiply(w).sum()
    moments = {}
    for r in range(u.shape[1]):
        g = np.array([[u2[r], wu2[r], uwu[r]]]).T / n
        G = (
            np.array(
                [
                    [2 * uwu[r], -wu2[r], n],
                    [2 * wwuwu[r], -wwu2[r], trWtW],
                    [uwwu[r] + wu2[r], -wwuwu[r], 0.0],
                ]
            )
            / n
        )
        moments[r] = [G, g]
    return moments


class SURerrorGM(BaseSURerrorGM, REGI.Regimes_Frame):
    """
    User class for SUR Error estimation by Maximum Likelihood

    Parameters
    ----------
    bigy         : dictionary
                   with vectors of dependent variable, one for
                   each equation
    bigX         : dictionary
                   with matrices of explanatory variables,
                   one for each equation
    w            : spatial weights object
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    nonspat_diag : boolean
                   flag for non-spatial diagnostics, default = False
    spat_diag    : boolean
                   flag for spatial diagnostics, default = False (to be implemented)
    vm           : boolean
                   flag for asymptotic variance for lambda and Sigma,
                   default = False (to be implemented)
    name_bigy    : dictionary
                   with name of dependent variable for each equation.
                   default = None, but should be specified is done when
                   sur_stackxy is used
    name_bigX    : dictionary
                   with names of explanatory variables for each equation.
                   default = None, but should be specified is done when
                   sur_stackxy is used
    name_ds      : string
                   name for the data set
    name_w       : string
                   name for the weights file
    name_regimes : string
                   name of regime variable for use in the output

    Attributes
    ----------
    n          : int
                 number of observations in each cross-section
    n_eq       : int
                 number of equations
    bigy       : dictionary
                 with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary
                 with matrices of explanatory variables,
                 one for each equation
    bigK       : array
                 n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : dictionary
                 spatially lagged dependent variable
    bigXlag    : dictionary
                 spatially lagged explanatory variable
    lamsur     : float
                 spatial autoregressive coefficient in ML SUR Error
    bSUR       : array
                 beta coefficients in ML SUR Error
    varb       : array
                 variance of beta coefficients in ML SUR Error
    sig        : array
                 error variance-covariance matrix in ML SUR Error
    bigE       : array
                 n by n_eq matrix of vectors of residuals for each equation
    sur_inf    : array
                 inference for regression coefficients, stand. error, t, p
    surchow    : array
                 list with tuples for Chow test on regression coefficients.
                 each tuple contains test value, degrees of freedom, p-value
    name_bigy  : dictionary
                 with name of dependent variable for each equation
    name_bigX  : dictionary
                 with names of explanatory variables for each
                 equation
    name_ds    : string
                 name for the data set
    name_w     : string
                 name for the weights file
    name_regimes : string
                   name of regime variable for use in the output


    Examples
    --------

    First import libpysal to load the spatial analysis tools.

    >>> import libpysal
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Queen
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format

    Open data on NCOVR US County Homicides (3085 areas) using pysal.open().
    This is the DBF associated with the NAT shapefile. Note that pysal.open()
    also reads data in CSV format.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path('natregimes.dbf'), 'r')

    The specification of the model to be estimated can be provided as lists.
    Each equation should be listed separately. Equation 1 has HR80 as dependent
    variable, and PS80 and UE80 as exogenous regressors.
    For equation 2, HR90 is the dependent variable, and PS90 and UE90 the
    exogenous regressors.

    >>> y_var = ['HR80','HR90']
    >>> x_var = [['PS80','UE80'],['PS90','UE90']]
    >>> yend_var = [['RD80'],['RD90']]
    >>> q_var = [['FP79'],['FP89']]

    The SUR method requires data to be provided as dictionaries. PySAL
    provides the tool sur_dictxy to create these dictionaries from the
    list of variables. The line below will create four dictionaries
    containing respectively the dependent variables (bigy), the regressors
    (bigX), the dependent variables' names (bigyvars) and regressors' names
    (bigXvars). All these will be created from th database (db) and lists
    of variables (y_var and x_var) created above.

    >>> bigy,bigX,bigyvars,bigXvars = spreg.sur_dictxy(db,y_var,x_var)

    To run a spatial error model, we need to specify the spatial weights matrix.
    To do that, we can open an already existing gal file or create a new one.
    In this example, we will create a new one from NAT.shp and transform it to
    row-standardized.

    >>> w = Queen.from_shapefile(nat.get_path("natregimes.shp"))
    >>> w.transform='r'

    We can now run the regression and then have a summary of the output by typing:
    print(reg.summary)

    Alternatively, we can just check the betas and standard errors, asymptotic t
    and p-value of the parameters:

    >>> reg = spreg.SURerrorGM(bigy,bigX,w=w,name_bigy=bigyvars,name_bigX=bigXvars,name_ds="NAT",name_w="nat_queen")
    >>> reg.bSUR
    {0: array([[3.97746866],
           [0.89021219],
           [0.43050363]]), 1: array([[2.93679119],
           [1.11002826],
           [0.48761542]])}
    >>> reg.sur_inf
    {0: array([[ 0.37251476, 10.67734504,  0.        ],
           [ 0.14224297,  6.25839153,  0.        ],
           [ 0.04322388,  9.95985608,  0.        ]]), 1: array([[ 0.33694902,  8.71583245,  0.        ],
           [ 0.13413626,  8.27537783,  0.        ],
           [ 0.04033105, 12.09032288,  0.        ]])}
    """

    def __init__(
        self,
        bigy,
        bigX,
        w,
        regimes=None,
        nonspat_diag=True,
        spat_diag=False,
        vm=False,
        name_bigy=None,
        name_bigX=None,
        name_ds=None,
        name_w=None,
        name_regimes=None,
    ):

        # check on variable names for listing results
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        # initialize names - should be generated by sur_stack
        self.n_eq = len(bigy.keys())
        if name_bigy:
            self.name_bigy = name_bigy
        else:  # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = "dep_var_" + str(r)
                self.name_bigy[r] = yn
        if name_bigX is None:
            name_bigX = {}
            for r in range(self.n_eq):
                k = bigX[r].shape[1] - 1
                name_x = ["var_" + str(i + 1) + "_" + str(r + 1) for i in range(k)]
                ct = "Constant_" + str(r + 1)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                name_bigX[r] = name_x

        if regimes is not None:
            self.constant_regi = "many"
            self.cols2regi = "all"
            self.regime_err_sep = False
            self.name_regimes = USER.set_name_ds(name_regimes)
            self.regimes_set = REGI._get_regimes_set(regimes)
            self.regimes = regimes
            cols2regi_dic = {}
            self.name_bigX = {}
            self.name_x_r = name_bigX

            for r in range(self.n_eq):
                cols2regi_dic[r] = REGI.check_cols2regi(
                    self.constant_regi, self.cols2regi, bigX[r], add_cons=False
                )
                USER.check_regimes(self.regimes_set, bigy[0].shape[0], bigX[r].shape[1])
                bigX[r], self.name_bigX[r] = REGI.Regimes_Frame.__init__(
                    self,
                    bigX[r],
                    regimes,
                    constant_regi=None,
                    cols2regi=cols2regi_dic[r],
                    names=name_bigX[r],
                )
        else:
            self.name_bigX = name_bigX

        BaseSURerrorGM.__init__(self, bigy=bigy, bigX=bigX, w=w)

        # inference
        self.sur_inf = sur_setp(self.bSUR, self.varb)

        # test on constancy of regression coefficients across equations
        if check_k(self.bigK):  # only for equal number of variables
            self.surchow = sur_chow(self.n_eq, self.bigK, self.bSUR, self.varb)
        else:
            self.surchow = None

        # listing of results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR) - GM SPATIAL ERROR MODEL"

        if regimes is not None:
            self.title = "SUR - GM SPATIAL ERROR MODEL WITH REGIMES"
            self.chow_regimes = {}
            varb_counter = 0
            for r in range(self.n_eq):
                counter_end = varb_counter + self.bSUR[r].shape[0]
                self.chow_regimes[r] = REGI._chow_run(
                    len(cols2regi_dic[r]),
                    0,
                    0,
                    len(self.regimes_set),
                    self.bSUR[r],
                    self.varb[varb_counter:counter_end, varb_counter:counter_end],
                )
                varb_counter = counter_end
            regimes = True

        SUMMARY.SUR(
            reg=self,
            nonspat_diag=nonspat_diag,
            spat_diag=spat_diag,
            lambd=True,
            ml=False,
            regimes=regimes,
        )


class BaseSURerrorML:
    """
    Base class for SUR Error estimation by Maximum Likelihood

    requires: scipy.optimize.minimize_scalar and scipy.optimize.minimize

    Parameters
    ----------
    bigy       : dictionary
                 with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary
                 with matrices of explanatory variables,
                 one for each equation
    w          : spatial weights object
    epsilon    : float
                 convergence criterion for ML iterations
                 default 0.0000001

    Attributes
    ----------
    n          : int
                 number of observations in each cross-section
    n2         : int
                 n/2
    n_eq       : int
                 number of equations
    bigy       : dictionary
                 with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary
                 with matrices of explanatory variables,
                 one for each equation
    bigK       : array
                 n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : dictionary
                 spatially lagged dependent variable
    bigXlag    : dictionary
                 spatially lagged explanatory variable
    lamols     : array
                 spatial autoregressive coefficients from equation by
                 equation ML-Error estimation
    clikerr    : float
                 concentrated log-likelihood from equation by equation
                 ML-Error estimation (no constant)
    bSUR0      : array
                 SUR estimation for betas without spatial autocorrelation
    llik       : float
                 log-likelihood for classic SUR estimation (includes constant)
    lamsur     : float
                 spatial autoregressive coefficient in ML SUR Error
    bSUR       : array
                 beta coefficients in ML SUR Error
    varb       : array
                 variance of beta coefficients in ML SUR Error
    sig        : array
                 error variance-covariance matrix in ML SUR Error
    corr       : array
                 error correlation matrix
    bigE       : array
                 n by n_eq matrix of vectors of residuals for each equation
    cliksurerr : float
                 concentrated log-likelihood from ML SUR Error (no constant)

    """

    def __init__(self, bigy, bigX, w, epsilon=0.0000001):
        # setting up constants
        self.n = w.n
        self.n2 = self.n / 2.0
        self.n_eq = len(bigy.keys())
        WS = w.sparse
        I = sp.identity(self.n)
        # variables
        self.bigy = bigy
        self.bigX = bigX
        # number of variables by equation
        self.bigK = np.zeros((self.n_eq, 1), dtype=np.int_)
        for r in range(self.n_eq):
            self.bigK[r] = self.bigX[r].shape[1]
        # spatially lagged variables
        self.bigylag = {}
        for r in range(self.n_eq):
            self.bigylag[r] = WS * self.bigy[r]
        # note: unlike WX as instruments, this includes the constant
        self.bigXlag = {}
        for r in range(self.n_eq):
            self.bigXlag[r] = WS * self.bigX[r]

        # spatial parameter starting values
        lam = np.zeros((self.n_eq, 1))  # initialize as an array
        fun0 = 0.0
        fun1 = 0.0
        for r in range(self.n_eq):
            res = minimize_scalar(
                err_c_loglik_sp,
                0.0,
                bounds=(-1.0, 1.0),
                args=(
                    self.n,
                    self.bigy[r],
                    self.bigylag[r],
                    self.bigX[r],
                    self.bigXlag[r],
                    I,
                    WS,
                ),
                method="bounded",
                options={"xatol": epsilon},
            )
            lam[r] = res.x
            fun1 += res.fun
        self.lamols = lam
        self.clikerr = -fun1  # negative because use in min

        # SUR starting values
        reg0 = BaseSUR(self.bigy, self.bigX, iter=True)
        bigE = reg0.bigE
        self.bSUR0 = reg0.bSUR
        self.llik = reg0.llik  # as is, includes constant

        # iteration
        lambdabounds = [(-1.0, +1.0) for i in range(self.n_eq)]
        while abs(fun0 - fun1) > epsilon:
            fun0 = fun1
            sply = filter_dict(lam, self.bigy, self.bigylag)
            splX = filter_dict(lam, self.bigX, self.bigXlag)
            WbigE = WS * bigE
            splbigE = bigE - WbigE * lam.T
            splXX, splXy = sur_crossprod(splX, sply)
            b1, varb1, sig1 = sur_est(splXX, splXy, splbigE, self.bigK)
            bigE = sur_resids(self.bigy, self.bigX, b1)
            res = minimize(
                clik,
                lam,
                args=(self.n, self.n2, self.n_eq, bigE, I, WS),
                method="L-BFGS-B",
                bounds=lambdabounds,
            )
            lam = res.x
            lam.resize((self.n_eq, 1))
            fun1 = res.fun
        self.lamsur = lam
        self.bSUR = b1
        self.varb = varb1
        self.sig = sig1
        self.corr = sur_corr(self.sig)
        self.bigE = bigE
        self.cliksurerr = -fun1  # negative because use in min, no constant


class SURerrorML(BaseSURerrorML, REGI.Regimes_Frame):
    """
    User class for SUR Error estimation by Maximum Likelihood

    Parameters
    ----------
    bigy         : dictionary
                   with vectors of dependent variable, one for
                   each equation
    bigX         : dictionary
                   with matrices of explanatory variables,
                   one for each equation
    w            : spatial weights object
    regimes      : list
                   default = None.
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    epsilon      : float
                   convergence criterion for ML iterations.
                   default 0.0000001
    nonspat_diag : boolean
                   flag for non-spatial diagnostics, default = True
    spat_diag    : boolean
                   flag for spatial diagnostics, default = False
    vm           : boolean
                   flag for asymptotic variance for lambda and Sigma,
                   default = False
    name_bigy    : dictionary
                   with name of dependent variable for each equation.
                   default = None, but should be specified is done when
                   sur_stackxy is used
    name_bigX    : dictionary
                   with names of explanatory variables for each equation.
                   default = None, but should be specified is done when
                   sur_stackxy is used
    name_ds      : string
                   name for the data set
    name_w       : string
                   name for the weights file
    name_regimes : string
                   name of regime variable for use in the output

    Attributes
    ----------
    n          : int
                 number of observations in each cross-section
    n2         : int
                 n/2
    n_eq       : int
                 number of equations
    bigy       : dictionary
                 with vectors of dependent variable, one for
                 each equation
    bigX       : dictionary
                 with matrices of explanatory variables,
                 one for each equation
    bigK       : array
                 n_eq x 1 array with number of explanatory variables
                 by equation
    bigylag    : dictionary
                 spatially lagged dependent variable
    bigXlag    : dictionary
                 spatially lagged explanatory variable
    lamols     : array
                 spatial autoregressive coefficients from equation by
                 equation ML-Error estimation
    clikerr    : float
                 concentrated log-likelihood from equation by equation
                 ML-Error estimation (no constant)
    bSUR0      : array
                 SUR estimation for betas without spatial autocorrelation
    llik       : float
                 log-likelihood for classic SUR estimation (includes constant)
    lamsur     : float
                 spatial autoregressive coefficient in ML SUR Error
    bSUR       : array
                 beta coefficients in ML SUR Error
    varb       : array
                 variance of beta coefficients in ML SUR Error
    sig        : array
                 error variance-covariance matrix in ML SUR Error
    bigE       : array
                 n by n_eq matrix of vectors of residuals for each equation
    cliksurerr : float
                 concentrated log-likelihood from ML SUR Error (no constant)
    sur_inf    : array
                 inference for regression coefficients, stand. error, t, p
    errllik    : float
                 log-likelihood for error model without SUR (with constant)
    surerrllik : float
                 log-likelihood for SUR error model (with constant)
    lrtest     : tuple
                 likelihood ratio test for off-diagonal Sigma elements
    likrlambda : tuple
                 likelihood ratio test on spatial autoregressive coefficients
    vm         : array
                 asymptotic variance matrix for lambda and Sigma (only for vm=True)
    lamsetp    : array
                 inference for lambda, stand. error, t, p (only for vm=True)
    lamtest    : tuple
                 with test for constancy of lambda across equations
                 (test value, degrees of freedom, p-value)
    joinlam    : tuple
                 with test for joint significance of lambda across
                 equations (test value, degrees of freedom, p-value)
    surchow    : list
                 with tuples for Chow test on regression coefficients.
                 each tuple contains test value, degrees of freedom, p-value
    name_bigy  : dictionary
                 with name of dependent variable for each equation
    name_bigX  : dictionary
                 with names of explanatory variables for each
                 equation
    name_ds    : string
                 name for the data set
    name_w     : string
                 name for the weights file
    name_regimes : string
                   name of regime variable for use in the output


    Examples
    --------

    First import libpysal to load the spatial analysis tools.

    >>> import libpysal
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Queen
    >>> import spreg
    >>> np.set_printoptions(suppress=True) #prevent scientific format

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile. Note that libpysal.io.open()
    also reads data in CSV format.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path('natregimes.dbf'), 'r')

    The specification of the model to be estimated can be provided as lists.
    Each equation should be listed separately. Equation 1 has HR80 as dependent
    variable, and PS80 and UE80 as exogenous regressors.
    For equation 2, HR90 is the dependent variable, and PS90 and UE90 the
    exogenous regressors.

    >>> y_var = ['HR80','HR90']
    >>> x_var = [['PS80','UE80'],['PS90','UE90']]
    >>> yend_var = [['RD80'],['RD90']]
    >>> q_var = [['FP79'],['FP89']]

    The SUR method requires data to be provided as dictionaries. PySAL
    provides the tool sur_dictxy to create these dictionaries from the
    list of variables. The line below will create four dictionaries
    containing respectively the dependent variables (bigy), the regressors
    (bigX), the dependent variables' names (bigyvars) and regressors' names
    (bigXvars). All these will be created from th database (db) and lists
    of variables (y_var and x_var) created above.

    >>> bigy,bigX,bigyvars,bigXvars = spreg.sur_dictxy(db,y_var,x_var)

    To run a spatial error model, we need to specify the spatial weights matrix.
    To do that, we can open an already existing gal file or create a new one.
    In this example, we will create a new one from NAT.shp and transform it to
    row-standardized.

    >>> w = Queen.from_shapefile(nat.get_path("natregimes.shp"))
    >>> w.transform='r'

    We can now run the regression and then have a summary of the output by typing:
    print(reg.summary)

    Alternatively, we can just check the betas and standard errors, asymptotic t
    and p-value of the parameters:

    >>> reg = spreg.SURerrorML(bigy,bigX,w=w,name_bigy=bigyvars,name_bigX=bigXvars,name_ds="NAT",name_w="nat_queen")
    >>> reg.bSUR
    {0: array([[4.02228606],
           [0.88489637],
           [0.42402845]]), 1: array([[3.04923031],
           [1.10972632],
           [0.47075678]])}

    >>> reg.sur_inf
    {0: array([[ 0.36692175, 10.96224484,  0.        ],
           [ 0.14129077,  6.26294545,  0.        ],
           [ 0.04267954,  9.93516909,  0.        ]]), 1: array([[ 0.33139967,  9.20106629,  0.        ],
           [ 0.13352591,  8.31094381,  0.        ],
           [ 0.04004097, 11.75687747,  0.        ]])}

    """

    def __init__(
        self,
        bigy,
        bigX,
        w,
        regimes=None,
        nonspat_diag=True,
        spat_diag=False,
        vm=False,
        epsilon=0.0000001,
        name_bigy=None,
        name_bigX=None,
        name_ds=None,
        name_w=None,
        name_regimes=None,
    ):

        # need checks on match between bigy, bigX dimensions
        # check on variable names for listing results
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        self.n_eq = len(bigy.keys())
        # initialize names - should be generated by sur_stack
        if name_bigy:
            self.name_bigy = name_bigy
        else:  # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = "dep_var_" + str(r)
                self.name_bigy[r] = yn
        if name_bigX is None:
            name_bigX = {}
            for r in range(self.n_eq):
                k = bigX[r].shape[1] - 1
                name_x = ["var_" + str(i + 1) + "_" + str(r + 1) for i in range(k)]
                ct = "Constant_" + str(r + 1)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                name_bigX[r] = name_x

        if regimes is not None:
            self.constant_regi = "many"
            self.cols2regi = "all"
            self.regime_err_sep = False
            self.name_regimes = USER.set_name_ds(name_regimes)
            self.regimes_set = REGI._get_regimes_set(regimes)
            self.regimes = regimes
            self.name_x_r = name_bigX
            cols2regi_dic = {}
            self.name_bigX = {}
            for r in range(self.n_eq):
                cols2regi_dic[r] = REGI.check_cols2regi(
                    self.constant_regi, self.cols2regi, bigX[r], add_cons=False
                )
                USER.check_regimes(self.regimes_set, bigy[0].shape[0], bigX[r].shape[1])
                bigX[r], self.name_bigX[r] = REGI.Regimes_Frame.__init__(
                    self,
                    bigX[r],
                    regimes,
                    constant_regi=None,
                    cols2regi=cols2regi_dic[r],
                    names=name_bigX[r],
                )
        else:
            self.name_bigX = name_bigX

        # moved init here
        BaseSURerrorML.__init__(self, bigy=bigy, bigX=bigX, w=w, epsilon=epsilon)

        # inference
        self.sur_inf = sur_setp(self.bSUR, self.varb)

        # adjust concentrated log lik for constant
        const = -self.n2 * (self.n_eq * (1.0 + np.log(2.0 * np.pi)))
        self.errllik = const + self.clikerr
        self.surerrllik = const + self.cliksurerr

        # LR test on off-diagonal sigma
        if nonspat_diag:
            M = self.n_eq * (self.n_eq - 1) / 2.0
            likrodiag = 2.0 * (self.surerrllik - self.errllik)
            plik1 = stats.chisqprob(likrodiag, M)
            self.lrtest = (likrodiag, int(M), plik1)
        else:
            self.lrtest = None

        # LR test on spatial autoregressive coefficients
        if spat_diag:
            liklambda = 2.0 * (self.surerrllik - self.llik)
            plik2 = stats.chisqprob(liklambda, self.n_eq)
            self.likrlambda = (liklambda, self.n_eq, plik2)
        else:
            self.likrlambda = None

        # asymptotic variance for spatial coefficient
        if vm:
            self.vm = surerrvm(self.n, self.n_eq, w, self.lamsur, self.sig)
            vlam = self.vm[: self.n_eq, : self.n_eq]
            self.lamsetp = lam_setp(self.lamsur, vlam)
            # test on constancy of lambdas
            R = REGI.buildR(kr=1, kf=0, nr=self.n_eq)
            w, p = REGI.wald_test(self.lamsur, R, np.zeros((R.shape[0], 1)), vlam)
            self.lamtest = (w, R.shape[0], p)
            if spat_diag:  # test on joint significance of lambdas
                Rj = np.identity(self.n_eq)
                wj, pj = REGI.wald_test(
                    self.lamsur, Rj, np.zeros((Rj.shape[0], 1)), vlam
                )
                self.joinlam = (wj, Rj.shape[0], pj)
            else:
                self.joinlam = None
        else:
            self.vm = None
            self.lamsetp = None
            self.lamtest = None
            self.joinlam = None

        # test on constancy of regression coefficients across equations
        if check_k(self.bigK):  # only for equal number of variables
            self.surchow = sur_chow(self.n_eq, self.bigK, self.bSUR, self.varb)
        else:
            self.surchow = None

        # listing of results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR) - ML SPATIAL ERROR MODEL"
        if regimes is not None:
            self.title = "SUR - ML SPATIAL ERROR MODEL - REGIMES"
            self.chow_regimes = {}
            varb_counter = 0
            for r in range(self.n_eq):
                counter_end = varb_counter + self.bSUR[r].shape[0]
                self.chow_regimes[r] = REGI._chow_run(
                    len(cols2regi_dic[r]),
                    0,
                    0,
                    len(self.regimes_set),
                    self.bSUR[r],
                    self.varb[varb_counter:counter_end, varb_counter:counter_end],
                )
                varb_counter = counter_end
            regimes = True

        SUMMARY.SUR(
            reg=self,
            nonspat_diag=nonspat_diag,
            spat_diag=spat_diag,
            lambd=True,
            regimes=regimes,
        )


def jacob(lam, n_eq, I, WS):
    """Log-Jacobian for SUR Error model

    Parameters
    ----------
    lam      : array
               n_eq by 1 array of spatial autoregressive parameters
    n_eq     : int
               number of equations
    I        : sparse matrix
               sparse Identity matrix
    WS       : sparse matrix
               sparse spatial weights matrix

    Returns
    -------
    logjac   : float
               the log Jacobian

    """
    logjac = 0.0
    for r in range(n_eq):
        lami = lam[r]
        lamWS = WS.multiply(lami)
        B = (I - lamWS).tocsc()
        LU = SuperLU(B)
        jj = np.sum(np.log(np.abs(LU.U.diagonal())))
        logjac += jj
    return logjac


def clik(lam, n, n2, n_eq, bigE, I, WS):
    """
    Concentrated (negative) log-likelihood for SUR Error model

    Parameters
    ----------
    lam         : array
                  n_eq x 1 array of spatial autoregressive parameters
    n           : int
                  number of observations in each cross-section
    n2          : int
                  n/2
    n_eq        : int
                  number of equations
    bigE        : array
                  n by n_eq matrix with vectors of residuals for
                  each equation
    I           : sparse Identity matrix
    WS          : sparse spatial weights matrix

    Returns
    -------
    -clik       : float
                  negative (for minimize) of the concentrated
                  log-likelihood function

    """
    WbigE = WS * bigE
    spfbigE = bigE - WbigE * lam.T
    sig = np.dot(spfbigE.T, spfbigE) / n
    ldet = la.slogdet(sig)[1]
    logjac = jacob(lam, n_eq, I, WS)
    clik = -n2 * ldet + logjac
    return -clik  # negative for minimize


def surerrvm(n, n_eq, w, lam, sig):
    """
    Asymptotic variance matrix for lambda and Sigma in
    ML SUR Error estimation

    Source: Anselin (1988) :cite:`Anselin1988`, Chapter 10.

    Parameters
    ----------
    n         : int
                number of cross-sectional observations
    n_eq      : int
                number of equations
    w         : spatial weights object
    lam       : array
                n_eq by 1 vector with spatial autoregressive coefficients
    sig       : array
                n_eq by n_eq matrix with cross-equation error covariances

    Returns
    -------
    vm        : array
                asymptotic variance-covariance matrix for spatial autoregressive
                coefficients and the upper triangular elements of Sigma
                n_eq + n_eq x (n_eq + 1) / 2 coefficients


    """
    # inverse Sigma
    sigi = la.inv(sig)
    sisi = sigi * sig
    # elements of Psi_lam,lam
    # trace terms
    trDi = np.zeros((n_eq, 1))
    trDDi = np.zeros((n_eq, 1))
    trDTDi = np.zeros((n_eq, 1))
    trDTiDj = np.zeros((n_eq, n_eq))
    WS = w.sparse
    I = sp.identity(n)
    for i in range(n_eq):
        lami = lam[i][0]
        lamWS = WS.multiply(lami)
        B = I - lamWS
        bb = B.todense()
        Bi = la.inv(bb)
        D = WS * Bi
        trDi[i] = np.trace(D)
        DD = np.dot(D, D)
        trDDi[i] = np.trace(DD)
        DD = np.dot(D.T, D)
        trDTDi[i] = np.trace(DD)
        for j in range(i + 1, n_eq):
            lamj = lam[j][0]
            lamWS = WS.multiply(lamj)
            B = I - lamWS
            bb = B.todense()
            Bi = la.inv(bb)
            Dj = WS * Bi
            DD = np.dot(D.T, Dj)
            trDTiDj[i, j] = np.trace(DD)
            trDTiDj[j, i] = trDTiDj[i, j]
    np.fill_diagonal(trDTiDj, trDTDi)

    sisjT = sisi * trDTiDj
    Vll = np.diagflat(trDDi) + sisjT

    # elements of Psi_lam_sig
    P = int(n_eq * (n_eq + 1) / 2)  # force ints to be ints
    tlist = [(i, j) for i in range(n_eq) for j in range(i, n_eq)]
    zog = sigi * trDi
    Vlsig = np.zeros((n_eq, P))
    for i in range(n_eq):
        for j in range(n_eq):
            if i > j:
                jj = tlist.index((j, i))
            else:
                jj = tlist.index((i, j))
            Vlsig[i, jj] = zog[i, j]

    # top of Psi
    vtop = np.hstack((Vll, Vlsig))

    # elements of Psi_sig_sig

    Vsig = np.zeros((P, P))
    for ij in range(P):
        i, j = tlist[ij]
        for hk in range(P):
            h, k = tlist[hk]
            if i == j:
                if h == k:
                    Vsig[ij, hk] = 0.5 * (sigi[i, h] ** 2)
                else:  # h not equal to k
                    Vsig[ij, hk] = sigi[i, h] * sigi[i, k]
            else:  # i not equal to j
                if h == k:
                    Vsig[ij, hk] = sigi[i, h] * sigi[j, h]
                else:  # h not equal to k
                    Vsig[ij, hk] = sigi[i, h] * sigi[j, k] + sigi[i, k] * sigi[j, h]
    Vsig = n * Vsig

    # bottom of Psi
    vbottom = np.hstack((Vlsig.T, Vsig))

    # all of Psi
    vbig = np.vstack((vtop, vbottom))

    # inverse of Psi
    vm = la.inv(vbig)

    return vm


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()
    import numpy as np
    import libpysal
    from .sur_utils import sur_dictxy, sur_dictZ
    from libpysal.examples import load_example
    from libpysal.weights import Queen

    nat = load_example("Natregimes")
    db = libpysal.io.open(nat.get_path("natregimes.dbf"), "r")
    y_var = ["HR80", "HR90"]
    x_var = [["PS80", "UE80"], ["PS90", "UE90"]]
    w = Queen.from_shapefile(nat.get_path("natregimes.shp"))
    w.transform = "r"
    bigy0, bigX0, bigyvars0, bigXvars0 = sur_dictxy(db, y_var, x_var)
    reg0 = SURerrorML(
        bigy0,
        bigX0,
        w,
        regimes=regimes,
        name_bigy=bigyvars0,
        name_bigX=bigXvars0,
        name_w="natqueen",
        name_ds="natregimes",
        vm=True,
        nonspat_diag=True,
        spat_diag=True,
    )

    #    reg0 = SURerrorGM(bigy0,bigX0,w,regimes=regimes,name_bigy=bigyvars0,name_bigX=bigXvars0,\
    #        name_w="natqueen",name_ds="natregimes",vm=False,nonspat_diag=True,spat_diag=False)

    print(reg0.summary)
