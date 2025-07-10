"""
Spatial Error Models with regimes module
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
import multiprocessing as mp
from . import regimes as REGI
from . import user_output as USER
from libpysal.weights.spatial_lag import lag_spatial
from .ols import BaseOLS
from .twosls import BaseTSLS
from .error_sp import BaseGM_Error, BaseGM_Endog_Error, _momentsGM_Error
from .utils import set_endog, iter_msg, sp_att, set_warn
from .utils import optim_moments, get_spFilter, get_lags
from .utils import spdot, RegressionPropsY
from .sputils import sphstack
import pandas as pd
from .output import output, _spat_pseudo_r2
from .error_sp_het_regimes import GM_Error_Het_Regimes, GM_Endog_Error_Het_Regimes, GM_Combo_Het_Regimes
from .error_sp_hom_regimes import GM_Error_Hom_Regimes, GM_Endog_Error_Hom_Regimes, GM_Combo_Hom_Regimes


class GM_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    """
    GMM method for a spatial error model with regimes, with results and diagnostics;
    based on Kelejian and Prucha (1998, 1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object
    constant_regi: string, optional
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    Always False, kept for consistency, ignored.
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    slx_vars     : either "all" (default) or list of booleans to select x variables
                   to be lagged       
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
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
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
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
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
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
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: string
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   *  'one': a vector of ones is appended to x and held constant across regimes

                   * 'many': a vector of ones is appended to x and considered different per regime
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import libpysal
    >>> import numpy as np
    >>> from libpysal.examples import load_example

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Error_Regimes
    >>> model = GM_Error_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print(model.output)
        var_names coefficients   std_err    zt_stat      prob
    0  0_CONSTANT     0.074811  0.379864   0.196942  0.843873
    1      0_PS90     0.786105  0.152315   5.161043       0.0
    2      0_UE90     0.538848  0.051942  10.373969       0.0
    3  1_CONSTANT     5.103761  0.471284   10.82949       0.0
    4      1_PS90     1.196009   0.19867   6.020074       0.0
    5      1_UE90     0.600532  0.057252  10.489217       0.0
    6      lambda       0.3641      None       None      None
    """

    def __init__(
        self,
        y,
        x,
        regimes,
        w,
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        constant_regi="many",
        cols2regi="all",
        regime_err_sep=False,
        regime_lag_sep=False,
        slx_lags=0,
        slx_vars="all",
        cores=False,
        name_ds=None,
        name_regimes=None,
        latex=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        x_constant, name_x, warn = USER.check_constant(
            x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)

        if slx_lags >0:
            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=False,
                                                slx_lags=slx_lags,slx_vars=slx_vars)
            set_warn(self,"WX is computed using the complete W, i.e. not trimmed by regimes.")

        self.name_x_r = USER.set_name_x(name_x, x_constant)

        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_w = USER.set_name_w(name_w, w)
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n
        self.y = y

        cols2regi = REGI.check_cols2regi(constant_regi, cols2regi, x_constant)
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x_constant.shape[1])
        self.regime_err_sep = regime_err_sep
        if regime_err_sep == True:
            if set(cols2regi) == set([True]):
                self._error_regimes_multi(
                    y, x_constant, regimes, w, slx_lags, cores, cols2regi, vm, name_x, latex
                )
            else:
                raise Exception(
                    "All coefficients must vary across regimes if regime_err_sep = True."
                )
        else:
            x_constant1 = sphstack(
                np.ones((x_constant.shape[0], 1)), x_constant)
            name_x = USER.set_name_x(name_x, x_constant, constant=True)
            self.x, self.name_x, xtype, x_rlist = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes,
                constant_regi=constant_regi,
                cols2regi=cols2regi[1:],
                names=name_x,
                rlist=True,
            )
            ols = BaseOLS(y=y, x=self.x)
            self.k = ols.x.shape[1]
            moments = _momentsGM_Error(w, ols.u)
            lambda1 = optim_moments(moments)
            xs = get_spFilter(w, lambda1, x_constant1)
            ys = get_spFilter(w, lambda1, y)
            xs = REGI.Regimes_Frame.__init__(
                self, xs, regimes, constant_regi=None, cols2regi=cols2regi
            )[0]
            ols2 = BaseOLS(y=ys, x=xs)

            # Output
            self.predy = spdot(self.x, ols2.betas)
            self.u = y - self.predy
            self.betas = np.vstack((ols2.betas, np.array([[lambda1]])))
            self.sig2 = ols2.sig2n
            self.e_filtered = self.u - lambda1 * lag_spatial(w, self.u)
            self.vm = self.sig2 * ols2.xtxi
            if slx_lags == 0:
                self.title = "GM SPATIALLY WEIGHTED MODEL - REGIMES"
            else:
                self.title = "GM SPATIALLY WEIGHTED MODEL + SLX (SLX-Error) - REGIMES"       
            self.name_x.append("lambda")
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}
            self.output = pd.DataFrame(self.name_x,
                                       columns=['var_names'])
            self.output['var_type'] = xtype + ['lambda']
            self.output['regime'] = x_rlist + ['_Global']
            self.output['equation'] = 0
            output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)

    def _error_regimes_multi(self, y, x, regimes, w, slx_lags, cores, cols2regi, vm, name_x, latex):
        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set
        )
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                results_p[r] = _work_error(*(y,x,regi_ids,r,w,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes))
                is_win = True
            else:
                pool = mp.Pool(cores)                
                results_p[r] = pool.apply_async(_work_error,args=(y,x,regi_ids,r,w,self.name_ds,self.name_y,name_x+['lambda'],self.name_w,self.name_regimes, ))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work_error,
                    args=(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w,
                        self.name_ds,
                        self.name_y,
                        name_x + ["lambda"],
                        self.name_w,
                        self.name_regimes,
                        #slx_lags,
                    ),
                )
            else:
                results_p[r] = _work_error(
                    *(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w,
                        self.name_ds,
                        self.name_y,
                        name_x + ["lambda"],
                        self.name_w,
                        self.name_regimes,
                        #slx_lags,
                    )
                )

        self.kryd = 0
        self.kr = len(cols2regi)
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * (self.kr + 1), 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()

        results = {}
        self.name_y, self.name_x = [], []
        self.output = pd.DataFrame(
            columns=['var_names', 'var_type', 'regime', 'equation'])
        counter = 0
        for r in self.regimes_set:
            """
            if is_win:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            """
            if not cores:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()

            self.vm[
                (counter * self.kr): ((counter + 1) * self.kr),
                (counter * self.kr): ((counter + 1) * self.kr),
            ] = results[r].vm
            self.betas[
                (counter * (self.kr + 1)): ((counter + 1) * (self.kr + 1)),
            ] = results[r].betas
            self.u[
                regi_ids[r],
            ] = results[r].u
            self.predy[
                regi_ids[r],
            ] = results[r].predy
            self.e_filtered[
                regi_ids[r],
            ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.output = pd.concat([self.output, pd.DataFrame({'var_names': results[r].name_x,
                                                                'var_type': ['x'] * (len(results[r].name_x) - 1) +
                                                                            ['lambda'],
                                                                'regime': r, 'equation': r})], ignore_index=True)
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class GM_Endog_Error_Regimes(RegressionPropsY, REGI.Regimes_Frame):

    """
    GMM method for a spatial error model with regimes and endogenous variables, with
    results and diagnostics; based on Kelejian and Prucha (1998,
    1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    yend         : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x)
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object
    constant_regi: string
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    Always False, kept for consistency, ignored.
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error type.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
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
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2         : float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regimes variable for use in output
    title         : string
                    Name of the regression method used
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : ['one', 'many']
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:

                    * 'one': a vector of ones is appended to x and held constant across regimes.

                    * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
                    Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import libpysal
    >>> import numpy as np
    >>> from libpysal.examples import load_example

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    For the endogenous models, we add the endogenous variable RD90 (resource deprivation)
    and we decide to instrument for it with FP89 (families below poverty):

    >>> yd_var = ['RD90']
    >>> yend = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify the spatial
    weights matrix that includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``NAT.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are all set with the preliminaries, we are good to run the model. In this
    case, we will need the variables (exogenous and endogenous), the
    instruments and the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Endog_Error_Regimes
    >>> model = GM_Endog_Error_Regimes(y, x, yend, q, regimes, w=w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT.dbf')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    endogenous variables included. Alternatively, we can have a summary of the
    output by typing: model.summary

    >>> print(model.output)
        var_names coefficients   std_err    zt_stat      prob
    0  0_CONSTANT     3.597178  0.522633   6.882796       0.0
    1      0_PS90     1.065203  0.137555   7.743852       0.0
    2      0_UE90      0.15822  0.063054   2.509282  0.012098
    6      0_RD90     2.461609  0.300711   8.185967       0.0
    3  1_CONSTANT     9.197542  0.473654  19.418268       0.0
    4      1_PS90     1.880815   0.18335  10.258046       0.0
    5      1_UE90    -0.248777  0.072786  -3.417919  0.000631
    7      1_RD90     3.579429  0.240413  14.888666       0.0
    8      lambda     0.255639      None       None      None
    """

    def __init__(
        self,
        y,
        x,
        yend,
        q,
        regimes,
        w,
        cores=False,
        vm=False,
        constant_regi="many",
        cols2regi="all",
        regime_err_sep=False,
        regime_lag_sep=False,
        slx_lags=0,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
        summ=True,
        add_lag=False,
        latex=False,
    ):

        n = USER.check_arrays(y, x, yend, q)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        x_constant, name_x, warn = USER.check_constant(
            x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)

        if slx_lags > 0:
            lag_x = get_lags(w, x_constant, slx_lags)
            x_constant = np.hstack((x_constant, lag_x))
            name_x += USER.set_name_spatial_lags(name_x, slx_lags)

        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_w = USER.set_name_w(name_w, w)
        self.n = n
        self.y = y

        if summ:
            name_yend = USER.set_name_yend(name_yend, yend)
            self.name_y = USER.set_name_y(name_y)
            name_q = USER.set_name_q(name_q, q)
        self.name_x_r = USER.set_name_x(name_x, x_constant) + name_yend

        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x_constant, yend=yend
        )
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x_constant.shape[1])
        self.regime_err_sep = regime_err_sep

        if regime_err_sep == True:
            if set(cols2regi) == set([True]):
                self._endog_error_regimes_multi(
                    y,
                    x_constant,
                    regimes,
                    w,
                    yend,
                    q,
                    slx_lags,
                    cores,
                    cols2regi,
                    vm,
                    name_x,
                    name_yend,
                    name_q,
                    add_lag,
                    latex,
                )
            else:
                raise Exception(
                    "All coefficients must vary across regimes if regime_err_sep = True."
                )
        else:
            x_constant1 = sphstack(
                np.ones((x_constant.shape[0], 1)), x_constant)
            name_x = USER.set_name_x(name_x, x_constant, constant=True)
            q, name_q, _ = REGI.Regimes_Frame.__init__(
                self, q, regimes, constant_regi=None, cols2regi="all", names=name_q
            )
            x, name_x, xtype, x_rlist = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes,
                constant_regi=constant_regi,
                cols2regi=cols2regi[1:],
                names=name_x,
                rlist=True,
            )
            yend2, name_yend, xtypeyd, yend_rlist = REGI.Regimes_Frame.__init__(
                self,
                yend,
                regimes,
                constant_regi=None,
                cols2regi=cols2regi,
                yend=True,
                names=name_yend,
                rlist=True,
            )

            tsls = BaseTSLS(y=y, x=x, yend=yend2, q=q)
            self.k = tsls.z.shape[1]
            self.x = tsls.x
            self.yend, self.z = tsls.yend, tsls.z
            moments = _momentsGM_Error(w, tsls.u)
            lambda1 = optim_moments(moments)
            xs = get_spFilter(w, lambda1, x_constant1)
            xs = REGI.Regimes_Frame.__init__(
                self, xs, regimes, constant_regi=None, cols2regi=cols2regi
            )[0]
            ys = get_spFilter(w, lambda1, y)
            yend_s = get_spFilter(w, lambda1, yend)
            yend_s = REGI.Regimes_Frame.__init__(
                self,
                yend_s,
                regimes,
                constant_regi=None,
                cols2regi=cols2regi,
                yend=True,
            )[0]
            tsls2 = BaseTSLS(ys, xs, yend_s, h=tsls.h)

            # Output
            self.betas = np.vstack((tsls2.betas, np.array([[lambda1]])))
            self.predy = spdot(tsls.z, tsls2.betas)
            self.u = y - self.predy
            self.sig2 = float(np.dot(tsls2.u.T, tsls2.u)) / self.n
            self.e_filtered = self.u - lambda1 * lag_spatial(w, self.u)
            self.vm = self.sig2 * tsls2.varb
            self.name_x = USER.set_name_x(name_x, x_constant, constant=True)
            self.name_yend = USER.set_name_yend(name_yend, yend)
            self.name_z = self.name_x + self.name_yend
            self.name_z.append("lambda")
            self.name_q = USER.set_name_q(name_q, q)
            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.kf += 1
            self.chow = REGI.Chow(self)
            self._cache = {}
            self.output = pd.DataFrame(self.name_z,
                                       columns=['var_names'])
            self.output['var_type'] = xtype + xtypeyd + ['lambda']
            self.output['regime'] = x_rlist + yend_rlist + ['_Global']
            self.output['equation'] = 0
            if summ:
                if slx_lags == 0:
                    self.title = ("GM SPATIALLY WEIGHTED 2SLS - REGIMES")
                else:
                    self.title = ("GM SPATIALLY WEIGHTED 2SLS WITH SLX (SLX-Error) - REGIMES")
                output(reg=self, vm=vm, robust=False,
                       other_end=False, latex=latex)

    def _endog_error_regimes_multi(
        self,
        y,
        x,
        regimes,
        w,
        yend,
        q,
        slx_lags,
        cores,
        cols2regi,
        vm,
        name_x,
        name_yend,
        name_q,
        add_lag,
        latex,
    ):

        regi_ids = dict(
            (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set
        )
        if add_lag != False:
            self.cols2regi += [True]
            cols2regi += [True]
            self.predy_e = np.zeros((self.n, 1), float)
            self.e_pred = np.zeros((self.n, 1), float)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                results_p[r] = _work_endog_error(*(y,x,yend,q,regi_ids,r,w,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag))
                is_win = True
            else:
                pool = mp.Pool(cores)        
                results_p[r] = pool.apply_async(_work_endog_error,args=(y,x,yend,q,regi_ids,r,w,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes,add_lag, ))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x + name_yend
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work_endog_error,
                    args=(
                        y,
                        x_constant,
                        yend,
                        q,
                        regi_ids,
                        r,
                        w,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        name_yend,
                        name_q,
                        self.name_w,
                        self.name_regimes,
                        add_lag,
                        slx_lags,
                    ),
                )
            else:
                results_p[r] = _work_endog_error(
                    *(
                        y,
                        x_constant,
                        yend,
                        q,
                        regi_ids,
                        r,
                        w,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        name_yend,
                        name_q,
                        self.name_w,
                        self.name_regimes,
                        add_lag,
                        slx_lags,
                    )
                )

        self.kryd, self.kf = 0, 0
        self.kr = len(cols2regi)
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * (self.kr + 1), 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.e_filtered = np.zeros((self.n, 1), float)
        """
        if not is_win:
            pool.close()
            pool.join()
        """
        if cores:
            pool.close()
            pool.join()

        results = {}
        (
            self.name_y,
            self.name_x,
            self.name_yend,
            self.name_q,
            self.name_z,
            self.name_h,
        ) = ([], [], [], [], [], [])
        counter = 0
        self.output = pd.DataFrame(
            columns=['var_names', 'var_type', 'regime', 'equation'])
        for r in self.regimes_set:
            """
            if is_win:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()
            """
            if not cores:
                results[r] = results_p[r]
            else:
                results[r] = results_p[r].get()

            self.vm[
                (counter * self.kr): ((counter + 1) * self.kr),
                (counter * self.kr): ((counter + 1) * self.kr),
            ] = results[r].vm
            self.betas[
                (counter * (self.kr + 1)): ((counter + 1) * (self.kr + 1)),
            ] = results[r].betas
            self.u[
                regi_ids[r],
            ] = results[r].u
            self.predy[
                regi_ids[r],
            ] = results[r].predy
            self.e_filtered[
                regi_ids[r],
            ] = results[r].e_filtered
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            if add_lag != False:
                self.predy_e[
                    regi_ids[r],
                ] = results[r].predy_e
                self.e_pred[
                    regi_ids[r],
                ] = results[r].e_pred
                results[r].other_top = _spat_pseudo_r2(results[r])
                v_type = ['o'] + ['x'] * (len(results[r].name_x)-1) + ['yend'] * \
                    (len(results[r].name_yend) - 1) + ['rho', 'lambda']
            else:
                results[r].other_top = ""
                v_type = ['o'] + ['x'] * (len(results[r].name_x)-1) + ['yend'] * \
                    len(results[r].name_yend) + ['lambda']
            self.output = pd.concat([self.output, pd.DataFrame({'var_names': results[r].name_z,
                                                                'var_type': v_type,
                                                                'regime': r, 'equation': r})], ignore_index=True)
            counter += 1
        self.chow = REGI.Chow(self)
        self.multi = results
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class GM_Combo_Regimes(GM_Endog_Error_Regimes, REGI.Regimes_Frame):

    """
    GMM method for a spatial lag and error model with regimes and endogenous
    variables, with results and diagnostics; based on Kelejian and Prucha (1998,
    1999) :cite:`Kelejian1998` :cite:`Kelejian1999`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (note:
                   this should not contain any variables from x)
    w            : pysal W object
                   Spatial weights object (always needed)
    constant_regi: string
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the GNSM type.       
    w_lags       : integer
                   Orders of W to include as instruments for the spatially
                   lagged dependent variable. For example, w_lags=1, then
                   instruments are WX; if w_lags=2, then WX, WWX; and so on.
    lag_q        : boolean
                   If True, then include spatial lags of the additional
                   instruments (q).
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
    betas        : array
                   kx1 array of estimated coefficients
    u            : array
                   nx1 array of residuals
    e_filtered   : array
                   nx1 array of spatially filtered residuals
    e_pred       : array
                   nx1 array of residuals (using reduced form)
    predy        : array
                   nx1 array of predicted y values
    predy_e      : array
                   nx1 array of predicted y values (using reduced form)
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2         : float
                   Sigma squared used in computations (based on filtered
                   residuals)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_yend     : list of strings
                    Names of endogenous variables for use in output
    name_z        : list of strings
                    Names of exogenous and endogenous variables for use in
                    output
    name_q        : list of strings
                    Names of external instruments
    name_h        : list of strings
                    Names of all instruments used in ouput
    name_w        : string
                    Name of weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regimes variable for use in output
    title         : string
                    Name of the regression method used
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each
                    observation to a regime. Assumed to be aligned with 'x'.
    constant_regi : string
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:

                    * 'one': a vector of ones is appended to x and held constant across regimes.

                    * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi     : list, 'all'
                    Ignored if regimes=False. Argument indicating whether each
                    column of x should be considered as different per regime
                    or held constant across regimes (False).
                    If a list, k booleans indicating for each variable the
                    option (True if one per regime, False to be held constant).
                    If 'all', all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
    kr            : int
                    Number of variables/columns to be "regimized" or subject
                    to change by regime. These will result in one parameter
                    estimate by regime for each variable (i.e. nr parameters per
                    variable)
    kf            : int
                    Number of variables/columns to be considered fixed or
                    global across regimes and hence only obtain one parameter
                    estimate
    nr            : int
                    Number of different regimes in the 'regimes' list
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial lag model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    The Combo class runs an SARAR model, that is a spatial lag+error model.
    In this case we will run a simple version of that, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GM_Combo_Regimes
    >>> model = GM_Combo_Regimes(y, x, regimes, w=w, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them. Note that because we are running the classical GMM error
    model from 1998/99, the spatial parameter is obtained as a point estimate, so
    although you get a value for it (there are for coefficients under
    model.betas), you cannot perform inference on it (there are only three
    values in model.se_betas). Also, this regression uses a two stage least
    squares estimation method that accounts for the endogeneity created by the
    spatial lag of the dependent variable. We can have a summary of the
    output by typing: model.summary
    Alternatively, we can check the betas:

    >>> print(model.output)
            var_names coefficients   std_err    zt_stat      prob
    0      0_CONSTANT     1.460707  0.704174   2.074356  0.038046
    1          0_PS90      0.95795  0.171485   5.586214       0.0
    2          0_UE90     0.565805  0.053665  10.543203       0.0
    3      1_CONSTANT     9.112998  1.525875   5.972311       0.0
    4          1_PS90      1.13382   0.20552    5.51683       0.0
    5          1_UE90      0.65169  0.061106  10.664938       0.0
    6  _Global_W_HR90    -0.458326  0.145599  -3.147859  0.001645
    7          lambda     0.613599      None       None      None

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. In this case we consider RD90 (resource deprivation)
    as an endogenous regressor.  We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GM_Combo_Regimes(y, x, regimes, yd, q, w=w, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT')
    >>> print(model.output)
            var_names coefficients   std_err    zt_stat      prob
    0      0_CONSTANT     3.419638  0.530676   6.443931       0.0
    1          0_PS90     1.040658  0.132714   7.841346       0.0
    2          0_UE90     0.166344   0.06058   2.745844  0.006036
    6          0_RD90      2.43014  0.289431   8.396263       0.0
    3      1_CONSTANT     8.865446  0.764064  11.603014       0.0
    4          1_PS90     1.851205  0.179698  10.301769       0.0
    5          1_UE90    -0.249085  0.071674  -3.475235   0.00051
    7          1_RD90     3.616455  0.253083  14.289586       0.0
    8  _Global_W_HR90     0.033087  0.061265   0.540057  0.589158
    9          lambda      0.18685      None       None      None
    """

    def __init__(
        self,
        y,
        x,
        regimes,
        yend=None,
        q=None,
        w=None,
        slx_lags=0,
        w_lags=1,
        lag_q=True,
        cores=False,
        constant_regi="many",
        cols2regi="all",
        regime_err_sep=False,
        regime_lag_sep=False,
        vm=False,
        name_y=None,
        name_x=None,
        name_yend=None,
        name_q=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
        latex=False,
    ):
        if regime_lag_sep and not regime_err_sep:
            set_warn(self, "regime_err_sep set to True when regime_lag_sep=True.")                
            regime_err_sep = True
        if regime_err_sep and not regime_lag_sep:
            set_warn(self, "regime_err_sep set to False when regime_lag_sep=False.")                
            regime_err_sep = False
        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        x_constant, name_x, warn = USER.check_constant(
            x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)
        self.name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)

        if regime_err_sep and any(col != True for col in cols2regi):
            set_warn(self, "All coefficients must vary across regimes if regime_err_sep = True, so setting cols2regi = 'all'.")
            cols2regi = "all"
            
        if slx_lags > 0:
            yend2, q2, wx = set_endog(y, x_constant, w, yend, q, w_lags, lag_q, slx_lags)
            x_constant = np.hstack((x_constant, wx))
            name_slx = USER.set_name_spatial_lags(name_x, slx_lags)
            name_q.extend(USER.set_name_q_sp(name_slx[-len(name_x):], w_lags, name_q, lag_q, force_all=True))
            name_x += name_slx   
            cols2regi = REGI.check_cols2regi(constant_regi, cols2regi, x_constant[:, :-1], yend=yend2, add_cons=False)
        else:
            name_q.extend(USER.set_name_q_sp(name_x, w_lags, name_q, lag_q, force_all=True))
            yend2, q2 = yend, q
            cols2regi = REGI.check_cols2regi(constant_regi, cols2regi, x_constant, yend=yend2, add_cons=False)

        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, n, x_constant.shape[1])
        self.regime_err_sep = regime_err_sep
        self.regime_lag_sep = regime_lag_sep

        if regime_lag_sep == True:
            if slx_lags == 0:
                add_lag = [w_lags, lag_q]
            else:
                add_lag = False
                cols2regi += [True]

        else:
            add_lag = False
            cols2regi += [False]
            if slx_lags == 0:
                yend2, q2 = set_endog(y, x_constant, w, yend2, q2, w_lags, lag_q)
            
        name_yend.append(USER.set_name_yend_sp(self.name_y))

        GM_Endog_Error_Regimes.__init__(
            self,
            y=y,
            x=x_constant,
            yend=yend2,
            q=q2,
            regimes=regimes,
            w=w,
            vm=vm,
            constant_regi=constant_regi,
            cols2regi=cols2regi,
            regime_err_sep=regime_err_sep,
            cores=cores,
            name_y=self.name_y,
            name_x=name_x,
            name_yend=name_yend,
            name_q=name_q,
            name_w=name_w,
            name_ds=name_ds,
            name_regimes=name_regimes,
            summ=False,
            add_lag=add_lag,
            latex=latex,
        )

        if regime_err_sep != True:
            self.rho = self.betas[-2]
            self.predy_e, self.e_pred, warn = sp_att(
                w, self.y, self.predy, yend2[:, -1].reshape(self.n, 1), self.rho
            )
            set_warn(self, warn)
            if slx_lags == 0:
                self.title = "SPATIALLY WEIGHTED 2SLS - GM-COMBO MODEL - REGIMES"
            else:
                self.title = "SPATIALLY WEIGHTED 2SLS - GM-COMBO WITH SLX (GNSM) - REGIMES"
            self.output.iat[-2,
                            self.output.columns.get_loc('var_type')] = 'rho'
            self.other_top = _spat_pseudo_r2(self)
            output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


class GMM_Error_Regimes(GM_Error_Regimes, GM_Combo_Regimes, GM_Endog_Error_Regimes,
                        GM_Error_Het_Regimes, GM_Combo_Het_Regimes, GM_Endog_Error_Het_Regimes, 
                        GM_Error_Hom_Regimes, GM_Combo_Hom_Regimes, GM_Endog_Error_Hom_Regimes
                        ):

    """
    Wrapper function to call any of the GM methods for a spatial error regimes model available in spreg

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    regimes      : list or pandas.Series
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    w            : pysal W object
                   Spatial weights object (always needed)
    yend         : array
                   Two dimensional array with n rows and one column for each
                   endogenous variable (if any)
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable to use as instruments (if any)
                   (note: this should not contain any variables from x)
    estimator    : string
                   Choice of estimator to be used. Options are: 'het', which
                   is robust to heteroskedasticity, 'hom', which assumes
                   homoskedasticity, and 'kp98', which does not provide
                   inference on the spatial parameter for the error term.
    constant_regi: string, optional
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    regime_lag_sep: boolean
                    Always False, kept for consistency, ignored.                   
    add_wy       : boolean
                   If True, then a spatial lag of the dependent variable is included.           
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX-Error or GNSM type.             
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output                   
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    hard_bound   : boolean
                   If true, raises an exception if the estimated spatial
                   autoregressive parameter is outside the maximum/minimum bounds.
    **kwargs     : keywords
                   Additional arguments to pass on to the estimators. 
                   See the specific functions for details on what can be used.

    Attributes
    ----------
    output       : dataframe
                   regression results pandas dataframe
    summary      : string
                   Summary of regression results and diagnostics (note: use in
                   conjunction with the print command)
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
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
    vm           : array
                   Variance covariance matrix (kxk)
    sig2         : float
                   Sigma squared used in computations
    std_err      : array
                   1xk array of standard errors of the betas
    z_stat       : list of tuples
                   z statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
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
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: string
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   *  'one': a vector of ones is appended to x and held constant across regimes

                   * 'many': a vector of ones is appended to x and considered different per regime
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_err_sep: boolean
                    If True, a separate regression is run for each regime.
    kr           : int
                   Number of variables/columns to be "regimized" or subject
                   to change by regime. These will result in one parameter
                   estimate by regime for each variable (i.e. nr parameters per
                   variable)
    kf           : int
                   Number of variables/columns to be considered fixed or
                   global across regimes and hence only obtain one parameter
                   estimate
    nr           : int
                   Number of different regimes in the 'regimes' list
    name_yend    : list of strings (optional)
                    Names of endogenous variables for use in output
    name_z       : list of strings (optional)
                    Names of exogenous and endogenous variables for use in
                    output
    name_q       : list of strings (optional)
                    Names of external instruments
    name_h       : list of strings (optional)
                    Names of all instruments used in ouput                   
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression    

    Examples
    --------

    We first need to import the needed modules, namely numpy to convert the
    data we read into arrays that ``spreg`` understands and ``libpysal`` to
    handle the weights and file management.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal.examples import load_example

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> nat = load_example('Natregimes')
    >>> db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it the
    dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = np.array([db.by_col(y_var)]).reshape(3085,1)

    Extract UE90 (unemployment rate) and PS90 (population structure) vectors from
    the DBF to be used as independent variables in the regression. Other variables
    can be inserted by adding their names to x_var, such as x_var = ['Var1','Var2','...]
    Note that PySAL requires this to be an nxj numpy array, where j is the
    number of independent variables (not including a constant). By default
    this model adds a vector of ones to the independent variables passed in.

    >>> x_var = ['PS90','UE90']
    >>> x = np.array([db.by_col(name) for name in x_var]).T

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = 'SOUTH'
    >>> regimes = db.by_col(r_var)

    Since we want to run a spatial error model, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations. To do that, we can open an already existing gal file or
    create a new one. In this case, we will create one from ``NAT.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    The GMM_Error_Regimes class can run error models and SARAR models, that is a spatial lag+error model.
    In this example we will run a simple version of the latter, where we have the
    spatial effects as well as exogenous variables. Since it is a spatial
    model, we have to pass in the weights matrix. If we want to
    have the names of the variables printed in the output summary, we will
    have to pass them in as well, although this is optional.

    >>> from spreg import GMM_Error_Regimes
    >>> model = GMM_Error_Regimes(y, x, regimes, w=w, add_wy=True, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')

    Once we have run the model, we can explore a little bit the output. The
    regression object we have created has many attributes so take your time to
    discover them.

    >>> print(model.output)
            var_names coefficients   std_err    zt_stat      prob
    0      0_CONSTANT     1.461317  0.848361   1.722517  0.084976
    1          0_PS90     0.958711  0.239834   3.997388  0.000064
    2          0_UE90     0.565825  0.063726   8.879088       0.0
    3      1_CONSTANT     9.115738  1.976874   4.611189  0.000004
    4          1_PS90     1.132419  0.334107   3.389387    0.0007
    5          1_UE90     0.651804  0.105518   6.177197       0.0
    6  _Global_W_HR90    -0.458677  0.180997  -2.534173  0.011271
    7          lambda     0.734354  0.035255  20.829823       0.0

    This class also allows the user to run a spatial lag+error model with the
    extra feature of including non-spatial endogenous regressors. This means
    that, in addition to the spatial lag and error, we consider some of the
    variables on the right-hand side of the equation as endogenous and we
    instrument for this. In this case we consider RD90 (resource deprivation)
    as an endogenous regressor.  We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> yd_var = ['RD90']
    >>> yd = np.array([db.by_col(name) for name in yd_var]).T
    >>> q_var = ['FP89']
    >>> q = np.array([db.by_col(name) for name in q_var]).T

    And then we can run and explore the model analogously to the previous combo:

    >>> model = GMM_Error_Regimes(y, x, regimes, yend=yd, q=q, w=w, add_wy=True, name_y=y_var, name_x=x_var, name_yend=yd_var, name_q=q_var, name_regimes=r_var, name_ds='NAT')
    >>> print(model.output)
            var_names coefficients   std_err    zt_stat      prob
    0      0_CONSTANT     1.461317  0.848361   1.722517  0.084976
    1          0_PS90     0.958711  0.239834   3.997388  0.000064
    2          0_UE90     0.565825  0.063726   8.879088       0.0
    3      1_CONSTANT     9.115738  1.976874   4.611189  0.000004
    4          1_PS90     1.132419  0.334107   3.389387    0.0007
    5          1_UE90     0.651804  0.105518   6.177197       0.0
    6  _Global_W_HR90    -0.458677  0.180997  -2.534173  0.011271
    7          lambda     0.734354  0.035255  20.829823       0.0

    The class also allows for estimating a GNS model by adding spatial lags of the exogenous variables, using the argument slx_lags:

    >>> model = GMM_Error_Regimes(y, x, regimes, w=w, add_wy=True, slx_lags=1, name_y=y_var, name_x=x_var, name_regimes=r_var, name_ds='NAT')
    >>> print(model.output)
             var_names coefficients   std_err   zt_stat      prob
    0       0_CONSTANT     0.192699  0.256922   0.75003  0.453237
    1           0_PS90     1.098019  0.232054  4.731743  0.000002
    2           0_UE90     0.606622   0.07762  7.815325       0.0
    3         0_W_PS90    -1.068778  0.203911 -5.241381       0.0
    4         0_W_UE90    -0.657932  0.176073  -3.73671  0.000186
    5       1_CONSTANT    -0.104299  1.790953 -0.058237   0.95356
    6           1_PS90     1.219796  0.316425  3.854936  0.000116
    7           1_UE90     0.678922  0.120491  5.634647       0.0
    8         1_W_PS90    -1.308599  0.536231 -2.440366  0.014672
    9         1_W_UE90    -0.708492  0.167057  -4.24102  0.000022
    10  _Global_W_HR90     1.033956  0.269252  3.840111  0.000123
    11          lambda    -0.384968  0.192256 -2.002366  0.045245
    

    """

    def __init__(
            self, y, x, regimes, w, yend=None, q=None, estimator='het', constant_regi="many", cols2regi="all", regime_err_sep=False,
            regime_lag_sep=False, add_wy=False, slx_lags=0, vm=False, name_y=None, name_x=None, name_w=None, name_regimes=None, name_yend=None,
            name_q=None, name_ds=None, latex=False, **kwargs):

        if estimator == 'het':
            if yend is None and not add_wy:
                GM_Error_Het_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                              name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error_Het_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                                    constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                                    name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif add_wy:
                GM_Combo_Het_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep, regime_lag_sep=regime_lag_sep,
                                              name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error_Regimes not allowed. Using default arguments instead.')
                GM_Error_Het_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                              name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex)
        elif estimator == 'hom':
            if yend is None and not add_wy:
                GM_Error_Hom_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                              name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error_Hom_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                                    constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                                    name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif add_wy:
                GM_Combo_Hom_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep, regime_lag_sep=regime_lag_sep,
                                              name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error_Regimes not allowed. Using default arguments instead.')
                GM_Error_Hom_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                              constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                              name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex)        
        elif estimator == 'kp98':
            if yend is None and not add_wy:
                GM_Error_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                          constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                          name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif yend is not None and not add_wy:
                GM_Endog_Error_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                                constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                                name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            elif add_wy:
                GM_Combo_Regimes.__init__(self, y=y, x=x, regimes=regimes, yend=yend, q=q, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                          constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep, regime_lag_sep=regime_lag_sep,
                                          name_yend=name_yend, name_q=name_q, name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex, **kwargs)
            else:
                set_warn(self, 'Combination of arguments passed to GMM_Error_Regimes not allowed. Using default arguments instead.')
                GM_Error_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                          constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                          name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex)     
        else:
            set_warn(self, 'Combination of arguments passed to GMM_Error_Regimes not allowed. Using default arguments instead.')
            GM_Error_Het_Regimes.__init__(self, y=y, x=x, regimes=regimes, w=w, slx_lags=slx_lags, vm=vm, name_y=name_y, name_x=name_x,
                                            constant_regi=constant_regi, cols2regi=cols2regi, regime_err_sep=regime_err_sep,
                                            name_w=name_w, name_regimes=name_regimes, name_ds=name_ds, latex=latex)

def _work_error(y, x, regi_ids, r, w, name_ds, name_y, name_x, name_w, name_regimes):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    model = BaseGM_Error(y_r, x_r, w_r.sparse)
    set_warn(model, warn)
    model.w = w_r
    model.title = "GM SPATIALLY WEIGHTED LEAST SQUARES ESTIMATION - REGIME %s" % r
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    return model


def _work_endog_error(
    y,
    x,
    yend,
    q,
    regi_ids,
    r,
    w,
    name_ds,
    name_y,
    name_x,
    name_yend,
    name_q,
    name_w,
    name_regimes,
    add_lag,
    slx_lags,
):
    w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    if yend is not None:
        yend_r = yend[regi_ids[r]]
        q_r = q[regi_ids[r]]
    else:
        yend_r, q_r = None, None
    if add_lag != False:
        yend_r, q_r = set_endog(
            y_r, x_r[:, 1:], w_r, yend_r, q_r, add_lag[0], add_lag[1]
        )
    model = BaseGM_Endog_Error(y_r, x_r, yend_r, q_r, w_r.sparse)
    set_warn(model, warn)
    if add_lag != False:
        model.rho = model.betas[-2]
        model.predy_e, model.e_pred, warn = sp_att(
            w_r, model.y, model.predy, model.yend[:, -
                                                  1].reshape(model.n, 1), model.rho
        )
        set_warn(model, warn)
    model.w = w_r
    if slx_lags == 0:
        if add_lag != False:
            model.title = "SPATIALLY WEIGHTED 2SLS - GM-COMBO MODEL - REGIME %s" % r            
        else:
            model.title = "SPATIALLY WEIGHTED 2SLS (GM) - REGIME %s" % r
    else:
        if add_lag != False:
            model.title = "GM SPATIAL COMBO MODEL + SLX (GNSM) - REGIME %s" % r            
        else:
            model.title = "GM SPATIALLY WEIGHTED 2SLS + SLX (SLX-Error) - REGIME %s" % r
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_yend = ["%s_%s" % (str(r), i) for i in name_yend]
    model.name_z = model.name_x + model.name_yend + [str(r)+"_lambda"]
    model.name_q = ["%s_%s" % (str(r), i) for i in name_q]
    model.name_h = model.name_x + model.name_q
    model.name_w = name_w
    model.name_regimes = name_regimes
    return model


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

    db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'), 'r')
    y = np.array(db.by_col("HOVAL"))
    y = np.reshape(y, (49, 1))
    X = []
    X.append(db.by_col("INC"))
    X = np.array(X).T
    yd = []
    yd.append(db.by_col("CRIME"))
    yd = np.array(yd).T
    q = []
    q.append(db.by_col("DISCBD"))
    q = np.array(q).T

    r_var = 'NSA'
    regimes = db.by_col(r_var)

    w = libpysal.weights.Rook.from_shapefile(
        libpysal.examples.get_path("columbus.shp"))
    w.transform = 'r'
    # reg = GM_Error_Regimes(y, X, regimes, w=w, name_x=['inc'], name_y='hoval', name_ds='columbus',
    #                       regime_err_sep=True)
    # reg = GM_Endog_Error_Regimes(y, X, yd, q, regimes, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'],
    #                         name_q=['discbd'], name_ds='columbus', regime_err_sep=True)
    reg = GM_Combo_Regimes(y, X, regimes, yd, q, w=w, name_x=['inc'], name_y='hoval', name_yend=['crime'],
                           name_q=['discbd'], name_ds='columbus', regime_err_sep=True, regime_lag_sep=True)
    print(reg.output)
    print(reg.summary)
