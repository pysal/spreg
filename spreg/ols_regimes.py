"""
Ordinary Least Squares regression with regimes.
"""

__author__ = "Luc Anselin, Pedro V. Amaral, Daniel Arribas-Bel"

import numpy as np
import multiprocessing as mp
import pandas as pd
from . import regimes as REGI
from . import user_output as USER
from .utils import set_warn, RegressionProps_basic, spdot, RegressionPropsY, get_lags, optim_k
from .ols import BaseOLS
from .robust import hac_multi
from .output import output, _spat_diag_out, _nonspat_mid, _nonspat_top
from .skater_reg import Skater_reg

class OLS_Regimes(BaseOLS, REGI.Regimes_Frame, RegressionPropsY):
    """
    Ordinary least squares with results and diagnostics.

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
                   Spatial weights object (required if running spatial
                   diagnostics)
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.  If 'hac', then a
                   HAC consistent estimator of the variance-covariance
                   matrix is given. Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    slx_lags       : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
                   Note: WX is computed using the complete weights matrix
    slx_vars     : either "All" (default) or list of booleans to select x variables
                   to be lagged                   
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    nonspat_diag : boolean
                   If True, then compute non-spatial diagnostics on
                   the regression.
    spat_diag    : boolean
                   If True, then compute Lagrange multiplier tests (requires
                   w). Note: see moran for further tests.
    moran        : boolean
                   If True, compute Moran's I on the residuals. Note:
                   requires spat_diag=True.
    white_test   : boolean
                   If True, compute White's specification robust test.
                   (requires nonspat_diag=True)
    vm           : boolean
                   If True, include variance-covariance matrix in summary
                   results
    constant_regi: string, optional
                   Switcher controlling the constant term setup. It may take
                   the following values:
                   *  'one': a vector of ones is appended to x and held constant across regimes
                   * 'many': a vector of ones is appended to x and considered different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    regime_err_sep  : boolean
                   If True, a separate regression is run for each regime.
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
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regime variable for use in the output
    latex        : boolean
                   Specifies if summary is to be printed in latex format

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
    robust       : string
                   Adjustment for robust standard errors
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    r2           : float
                   R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    ar2          : float
                   Adjusted R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2ML       : float
                   Sigma squared (maximum likelihood)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    f_stat       : tuple
                   Statistic (float), p-value (float)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   Log likelihood
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    aic          : float
                   Akaike information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    schwarz      : float
                   Schwarz information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    t_stat       : list of tuples
                   t statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mulColli     : float
                   Multicollinearity condition number
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    jarque_bera  : dictionary
                   'jb': Jarque-Bera statistic (float); 'pvalue': p-value
                   (float); 'df': degrees of freedom (int)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    breusch_pagan : dictionary
                    'bp': Breusch-Pagan statistic (float); 'pvalue': p-value
                    (float); 'df': degrees of freedom (int)
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    koenker_bassett: dictionary
                     'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float);
                     'df': degrees of freedom (int). Only available in dictionary
                     'multi' when multiple regressions (see 'multi' below for details).
    white         : dictionary
                    'wh': White statistic (float); 'pvalue': p-value (float);
                    'df': degrees of freedom (int).
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    lm_error      : tuple
                    Lagrange multiplier test for spatial error model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    lm_lag        : tuple
                    Lagrange multiplier test for spatial lag model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    rlm_error     : tuple
                    Robust lagrange multiplier test for spatial error model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    rlm_lag       : tuple
                    Robust lagrange multiplier test for spatial lag model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float. Only available in dictionary 'multi' when
                    multiple regressions (see 'multi' below for details)
    lm_sarma      : tuple
                    Lagrange multiplier test for spatial SARMA model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    moran_res     : tuple
                    Moran's I for the residuals; tuple containing the triple
                    (Moran's I, standardized Moran's I, p-value)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_w        : string
                    Name of weights matrix for use in output
    name_gwk      : string
                    Name of kernel weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regime variable for use in the output
    title         : string
                    Name of the regression method used.
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    sig2n         : float
                    Sigma squared (computed with n in the denominator)
    sig2n_k       : float
                    Sigma squared (computed with n-k in the denominator)
    xtx           : float
                    :math:`X'X`. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    xtxi          : float
                    :math:`(X'X)^{-1}`. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each observation to
                    a regime. Assumed to be aligned with 'x'.
    constant_regi : string
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:

                    *  'one': a vector of ones is appended to x and held constant across regimes.

                    * 'many': a vector of ones is appended to x and considered different per regime.
    cols2regi    : list
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
                   estimate.
    nr           : int
                   Number of different regimes in the 'regimes' list.
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression.

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import OLS_Regimes

    Open data on NCOVR US County Homicides (3085 areas) using libpysal.io.open().
    This is the DBF associated with the NAT shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path("NAT.dbf"),'r')

    Extract the HR90 column (homicide rates in 1990) from the DBF file and make it
    the dependent variable for the regression. Note that PySAL requires this to be
    an numpy array of shape (n, 1) as opposed to the also common shape of (n, )
    that other packages accept.

    >>> y_var = 'HR90'
    >>> y = db.by_col(y_var)
    >>> y = np.array(y)

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

    We can now run the regression and then have a summary of the output
    by typing: olsr.summary

    >>> olsr = OLS_Regimes(y, x, regimes, nonspat_diag=False, name_y=y_var, name_x=['PS90','UE90'], name_regimes=r_var, name_ds='NAT')
    >>> print(olsr.summary)
    REGRESSION RESULTS
    ------------------
    <BLANKLINE>
    SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES ESTIMATION - REGIME 0
    ---------------------------------------------------------------
    Data set            :         NAT
    Weights matrix      :        None
    Dependent Variable  :      0_HR90                Number of Observations:        1673
    Mean dependent var  :      3.3416                Number of Variables   :           3
    S.D. dependent var  :      4.6795                Degrees of Freedom    :        1670
    R-squared           :      0.1271
    Adjusted R-squared  :      0.1260
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     t-Statistic     Probability
    ------------------------------------------------------------------------------------
              0_CONSTANT         0.39643         0.24816         1.59745         0.11035
                  0_PS90         0.65583         0.09663         6.78728         0.00000
                  0_UE90         0.48704         0.03629        13.42213         0.00000
    ------------------------------------------------------------------------------------
    Regimes variable: SOUTH
    <BLANKLINE>
    SUMMARY OF OUTPUT: ORDINARY LEAST SQUARES ESTIMATION - REGIME 1
    ---------------------------------------------------------------
    Data set            :         NAT
    Weights matrix      :        None
    Dependent Variable  :      1_HR90                Number of Observations:        1412
    Mean dependent var  :      9.5493                Number of Variables   :           3
    S.D. dependent var  :      7.0389                Degrees of Freedom    :        1409
    R-squared           :      0.0661
    Adjusted R-squared  :      0.0647
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     t-Statistic     Probability
    ------------------------------------------------------------------------------------
              1_CONSTANT         5.59835         0.46895        11.93816         0.00000
                  1_PS90         1.16210         0.21667         5.36338         0.00000
                  1_UE90         0.53164         0.05946         8.94164         0.00000
    ------------------------------------------------------------------------------------
    Regimes variable: SOUTH
    ------------------------------------------------------------------------------------
    <BLANKLINE>
    GLOBAL DIAGNOSTICS
    <BLANKLINE>
    REGIMES DIAGNOSTICS - CHOW TEST
                     VARIABLE        DF        VALUE           PROB
                CONSTANT              1         96.129           0.0000
                    PS90              1          4.554           0.0328
                    UE90              1          0.410           0.5220
             Global test              3        680.960           0.0000
    ================================ END OF REPORT =====================================
    """

    def __init__(
            self,
            y,
            x,
            regimes,
            w=None,
            robust=None,
            gwk=None,
            slx_lags=0,
            slx_vars = "all",
            sig2n_k=True,
            nonspat_diag=True,
            spat_diag=False,
            moran=False,
            white_test=False,
            vm=False,
            constant_regi="many",
            cols2regi="all",
            regime_err_sep=True,
            cores=False,
            name_y=None,
            name_x=None,
            name_regimes=None,
            name_w=None,
            name_gwk=None,
            name_ds=None,
            latex=False,
            **kwargs,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        USER.check_robust(robust, gwk)
        if robust == "hac":
            if regime_err_sep:
                set_warn(
                    self,
                    "Error by regimes is not available for HAC estimation. The error by regimes has been disabled for this model.",
                )
                regime_err_sep = False
        spat_diag, warn = USER.check_spat_diag(spat_diag=spat_diag, w=w, robust=robust, slx_lags=slx_lags)
        set_warn(self, warn)
        if robust in ["hac", "white"] and white_test:
            set_warn(
                self,
                "White test not available when standard errors are estimated by HAC or White correction.",
            )
            white_test = False

        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)        
        if spat_diag or moran or slx_lags > 0:
            w = USER.check_weights(w, y, slx_lags=slx_lags, w_required=True, allow_wk=True)
        else:
            w = USER.check_weights(w, y, slx_lags=slx_lags, allow_wk=True)
        if slx_lags > 0:
            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=False,
                                                slx_lags=slx_lags,slx_vars=slx_vars)
            set_warn(self,"WX is computed using the complete W, i.e. not trimmed by regimes.")

        self.slx_lags = slx_lags
        self.name_x_r = USER.set_name_x(name_x, x_constant)
        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)

        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x_constant, add_cons=False
        )
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        USER.check_regimes(self.regimes_set, self.n, x_constant.shape[1])
        self.regime_err_sep = regime_err_sep
        if (
                regime_err_sep == True
                and set(cols2regi) == set([True])
                and constant_regi == "many"
        ):
            self.y = y
            regi_ids = dict(
                (r, list(np.where(np.array(regimes) == r)[0])) for r in self.regimes_set
            )
            self._ols_regimes_multi(
                x_constant,
                w,
                regi_ids,
                cores,
                gwk,
                slx_lags,
                sig2n_k,
                robust,
                nonspat_diag,
                spat_diag,
                vm,
                name_x,
                moran,
                white_test,
                latex
            )
        else:
            x, self.name_x, xtype, x_rlist = REGI.Regimes_Frame.__init__(
                self, x_constant, regimes, constant_regi, cols2regi, name_x, rlist=True
            )

            self.output = pd.DataFrame(self.name_x,
                                       columns=['var_names'])
            self.output['var_type'] = xtype
            self.output['regime'] = x_rlist
            self.output['equation'] = 0
            BaseOLS.__init__(self, y=y, x=x, robust=robust, gwk=gwk, sig2n_k=sig2n_k)
            if regime_err_sep == True and robust == None:
                y2, x2 = REGI._get_weighted_var(
                    regimes, self.regimes_set, sig2n_k, self.u, y, x
                )
                ols2 = BaseOLS(y=y2, x=x2, sig2n_k=sig2n_k)
                RegressionProps_basic(self, betas=ols2.betas, vm=ols2.vm)
                self.title = (
                    "ORDINARY LEAST SQUARES - REGIMES (Group-wise heteroskedasticity)"
                )
                if slx_lags > 0:
                    self.title = "ORDINARY LEAST SQUARES WITH SLX - REGIMES (Group-wise heteroskedasticity)"
                nonspat_diag = None
                set_warn(
                    self,
                    "Residuals treated as homoskedastic for the purpose of diagnostics.",
                )
            else:
                if slx_lags == 0:
                    self.title = "ORDINARY LEAST SQUARES - REGIMES"
                else:
                    self.title = "ORDINARY LEAST SQUARES WITH SLX - REGIMES"
            self.robust = USER.set_robust(robust)
            self.chow = REGI.Chow(self)
            self.other_mid, other_end = ("", "")  # strings where function-specific diag. are stored
            if nonspat_diag:
                self.other_mid += _nonspat_mid(self, white_test=white_test)
                top_diag = _nonspat_top(self)
                try:
                    self.other_top += top_diag
                except:
                    self.other_top = top_diag
            if spat_diag:
                other_end += _spat_diag_out(self, w, 'ols', moran=moran) #Must decide what to do with W.
            output(reg=self, vm=vm, robust=robust, other_end=other_end, latex=latex)

    def _ols_regimes_multi(
            self,
            x,
            w,
            regi_ids,
            cores,
            gwk,
            slx_lags,
            sig2n_k,
            robust,
            nonspat_diag,
            spat_diag,
            vm,
            name_x,
            moran,
            white_test,
            latex
    ):
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(self.y,x,w,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work,args=(self.y,x,w,regi_ids,r,robust,sig2n_k,self.name_ds,self.name_y,name_x,self.name_w,self.name_regimes))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work,
                    args=(
                        self.y,
                        x_constant,
                        w,
                        regi_ids,
                        r,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        self.name_w,
                        self.name_regimes,
                        slx_lags
                    ),
                )
            else:
                results_p[r] = _work(
                    *(
                        self.y,
                        x_constant,
                        w,
                        regi_ids,
                        r,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        self.name_w,
                        self.name_regimes,
                        slx_lags
                    )
                )
        self.kryd = 0
        self.kr = x_constant.shape[1]
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
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
        counter = 0
        self.output = pd.DataFrame(columns=['var_names', 'var_type', 'regime', 'equation'])
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
            (counter * self.kr): ((counter + 1) * self.kr),
            ] = results[r].betas
            self.u[
                regi_ids[r],
            ] = results[r].u
            self.predy[
                regi_ids[r],
            ] = results[r].predy
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.output = pd.concat([self.output, pd.DataFrame({'var_names': results[r].name_x,
                                                       'var_type': ['o']+['x']*(len(results[r].name_x)-1),
                                                       'regime': r, 'equation': r})], ignore_index=True)
            try:
                results[r].other_top = self.other_top
            except:
                results[r].other_top = ""
            results[r].other_mid = ""
            if nonspat_diag:
                results[r].other_mid += _nonspat_mid(results[r], white_test=white_test)
                results[r].other_top += _nonspat_top(results[r])
            counter += 1
        self.multi = results
        self.hac_var = x_constant[:, 1:]
        if robust == "hac":
            hac_multi(self, gwk)
        self.chow = REGI.Chow(self)
        other_end = ""
        if spat_diag:
            self._get_spat_diag_props(x_constant, sig2n_k)
            other_end += _spat_diag_out(self, w, 'ols', moran=moran) #Need to consider W 
        output(reg=self, vm=vm, robust=robust, other_end=other_end, latex=latex)

    def _get_spat_diag_props(self, x, sig2n_k):
        self.k = self.kr
        self._cache = {}
        self.x = REGI.regimeX_setup(
            x, self.regimes, [True] * x.shape[1], self.regimes_set
        )
        self.xtx = spdot(self.x.T, self.x)
        self.xtxi = np.linalg.inv(self.xtx)


def _work(
        y, x, w, regi_ids, r, robust, sig2n_k, name_ds, name_y, name_x, name_w, name_regimes, slx_lags
):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    # x_constant,name_x,warn = USER.check_constant(x_r, name_x)
    # name_x = USER.set_name_x(name_x, x_constant)
    if robust == "hac":
        robust = None
    model = BaseOLS(y_r, x_r, robust=robust, sig2n_k=sig2n_k)
    if slx_lags == 0:
        model.title = "ORDINARY LEAST SQUARES ESTIMATION - REGIME %s" % r
    else:
        model.title = "ORDINARY LEAST SQUARES ESTIMATION WITH SLX - REGIME %s" % r
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    if w:
        w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
        set_warn(model, warn)
        model.w = w_r
    return model


class OLS_Endog_Regimes(OLS_Regimes):
    """
    Ordinary least squares with endogenous regimes. Based on the function skater_reg as shown in :cite:`Anselin2021`.

    Parameters
    ----------
    y            : numpy.ndarray or pandas.Series
                   nx1 array for dependent variable
    x            : numpy.ndarray or pandas object
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, excluding the constant
    w            : pysal W object
                   Spatial weights object (required if running spatial
                   diagnostics)
    n_clusters   : int
                   Number of clusters to be used in the endogenous regimes.
                   If None (default), the number of clusters will be chosen
                   according to the function utils.optim_k using a method 
                   adapted from Mojena (1977)'s Rule Two
    quorum       : int
                   Minimum number of observations in a cluster to be considered
                   Must be at least larger than the number of variables in x
                   Default value is 30 or 10*k, whichever is larger.
    trace        : boolean
                   Sets  whether to store intermediate results of the clustering
                   Hard-coded to True if n_clusters is None
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    latex        : boolean
                   Specifies if summary is to be printed in latex format
    **kwargs     : additional keyword arguments depending on the specific model

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
    robust       : string
                   Adjustment for robust standard errors
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (kxk)
    r2           : float
                   R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    ar2          : float
                   Adjusted R squared
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2ML       : float
                   Sigma squared (maximum likelihood)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    f_stat       : tuple
                   Statistic (float), p-value (float)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   Log likelihood
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    aic          : float
                   Akaike information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    schwarz      : float
                   Schwarz information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    std_err      : array
                   1xk array of standard errors of the betas
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    t_stat       : list of tuples
                   t statistic; each tuple contains the pair (statistic,
                   p-value), where each is a float
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    mulColli     : float
                   Multicollinearity condition number
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    jarque_bera  : dictionary
                   'jb': Jarque-Bera statistic (float); 'pvalue': p-value
                   (float); 'df': degrees of freedom (int)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    breusch_pagan : dictionary
                    'bp': Breusch-Pagan statistic (float); 'pvalue': p-value
                    (float); 'df': degrees of freedom (int)
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    koenker_bassett: dictionary
                     'kb': Koenker-Bassett statistic (float); 'pvalue': p-value (float);
                     'df': degrees of freedom (int). Only available in dictionary
                     'multi' when multiple regressions (see 'multi' below for details).
    white         : dictionary
                    'wh': White statistic (float); 'pvalue': p-value (float);
                    'df': degrees of freedom (int).
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    lm_error      : tuple
                    Lagrange multiplier test for spatial error model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    lm_lag        : tuple
                    Lagrange multiplier test for spatial lag model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    rlm_error     : tuple
                    Robust lagrange multiplier test for spatial error model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    rlm_lag       : tuple
                    Robust lagrange multiplier test for spatial lag model;
                    tuple contains the pair (statistic, p-value), where each
                    is a float. Only available in dictionary 'multi' when
                    multiple regressions (see 'multi' below for details)
    lm_sarma      : tuple
                    Lagrange multiplier test for spatial SARMA model; tuple
                    contains the pair (statistic, p-value), where each is a
                    float. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    moran_res     : tuple
                    Moran's I for the residuals; tuple containing the triple
                    (Moran's I, standardized Moran's I, p-value)
    name_y        : string
                    Name of dependent variable for use in output
    name_x        : list of strings
                    Names of independent variables for use in output
    name_w        : string
                    Name of weights matrix for use in output
    name_gwk      : string
                    Name of kernel weights matrix for use in output
    name_ds       : string
                    Name of dataset for use in output
    name_regimes  : string
                    Name of regime variable for use in the output
    title         : string
                    Name of the regression method used.
                    Only available in dictionary 'multi' when multiple regressions
                    (see 'multi' below for details)
    sig2n         : float
                    Sigma squared (computed with n in the denominator)
    sig2n_k       : float
                    Sigma squared (computed with n-k in the denominator)
    xtx           : float
                    :math:`X'X`. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    xtxi          : float
                    :math:`(X'X)^{-1}`. Only available in dictionary 'multi' when multiple
                    regressions (see 'multi' below for details)
    regimes       : list
                    List of n values with the mapping of each observation to
                    a regime. Assumed to be aligned with 'x'.
    constant_regi : string
                    Ignored if regimes=False. Constant option for regimes.
                    Switcher controlling the constant term setup. It may take
                    the following values:

                    *  'one': a vector of ones is appended to x and held constant across regimes.

                    * 'many': a vector of ones is appended to x and considered different per regime.
    cols2regi    : list
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
                   estimate.
    nr           : int
                   Number of different regimes in the 'regimes' list.
    multi        : dictionary
                   Only available when multiple regressions are estimated,
                   i.e. when regime_err_sep=True and no variable is fixed
                   across regimes.
                   Contains all attributes of each individual regression.
    SSR          : list
                   list with the total sum of squared residuals for the model 
                   considering all regimes for each of steps of number of regimes
                   considered, starting with the solution with 2 regimes. 
    clusters     : int
                   Number of clusters considered in the endogenous regimes
    _trace       : list
                   List of dictionaries with the clustering results for each
                   number of clusters tested. Only available if n_clusters is
                   None or trace=True.

    Examples
    --------
    >>> import libpysal
    >>> import numpy as np
    >>> np.set_printoptions(legacy='1.25') #to avoid printing issues with numpy floats    
    >>> import geopandas as gpd
    >>> from spreg import OLS_Endog_Regimes

    Open data on Baltimore house sales price and characteristics in Baltimore
    from libpysal examples using geopandas.

    >>> db = gpd.read_file(libpysal.examples.get_path('baltim.shp'))

    We will create a weights matrix based on contiguity.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = "r"

    For this example, we will use the 'PRICE' column as the dependent variable and
    the 'NROOM', 'AGE', and 'SQFT' columns as independent variables.
    At this point, we will let the model choose the number of clusters.

    >>> olsr = OLS_Endog_Regimes(y=db['PRICE'], x=db[['NROOM','AGE','SQFT']], w=w, name_w="baltim_q.gal")
    
    The function `print(olsr.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes:
    >>> olsr.betas
    array([[26.24209866],
           [ 2.40329959],
           [-0.24183707],
           [ 0.45714794],
           [19.84817747],
           [ 5.12117483],
           [-0.65466516],
           [ 1.10034154]])
    >>> olsr.SSR
    [68840.74965798721, 62741.55717492997]
    >>> olsr.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int32)

    We  will now set the number of clusters to 2 and run the regression again.

    >>> olsr = OLS_Endog_Regimes(y=db['PRICE'], x=db[['NROOM','AGE','SQFT']], w=w, n_clusters=2, name_w="baltim_q.gal")

    The function `print(olsr.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes as before:
    >>> olsr.betas
    array([[26.24209866],
           [ 2.40329959],
           [-0.24183707],
           [ 0.45714794],
           [19.84817747],
           [ 5.12117483],
           [-0.65466516],
           [ 1.10034154]])
    >>> olsr.SSR
    [68840.74965798721]
    >>> olsr.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int32)

    """
   
    def __init__(
        self, y, x, w, n_clusters=None, quorum=-1, trace=True, name_y=None, name_x=None,
         constant_regi='many', cols2regi='all', regime_err_sep=True, **kwargs):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True)
        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        warn = USER.check_regi_args(constant_regi, cols2regi, regime_err_sep)
        set_warn(self, warn)
        # Standardize the variables
        x_std = (x_constant - np.mean(x_constant, axis=0)) / np.std(x_constant, axis=0)

        if quorum < 0:
            quorum = np.max([(x_constant.shape[1]+1)*10, 30])

        if not n_clusters:
            n_clusters_opt = x_constant.shape[0]*0.70//quorum
            if n_clusters_opt < 2:
                raise ValueError(
                    "The combination of the values of `N` and `quorum` is not compatible with regimes estimation.")
            sk_reg_results = Skater_reg().fit(n_clusters_opt, w, x_std, {'reg':BaseOLS,'y':y,'x':x_constant}, quorum=quorum, trace=True)
            n_clusters = optim_k([sk_reg_results._trace[i][1][2] for i in range(1, len(sk_reg_results._trace))])
            self.clusters = sk_reg_results._trace[n_clusters-1][0]
            self.score = sk_reg_results._trace[n_clusters-1][1][2]
        else:
            try:
                # Call the Skater_reg method based on OLS
                sk_reg_results = Skater_reg().fit(n_clusters, w, x_std, {'reg':BaseOLS,'y':y,'x':x_constant}, quorum=quorum, trace=trace)
                self.clusters = sk_reg_results.current_labels_
                self.score = sk_reg_results._trace[-1][1][2]
            except Exception as e:
                if str(e) == "one or more input arrays have more columns than rows":
                    raise ValueError("One or more input ended up with more variables than observations. Please check your setting for `quorum`.")
                else:
                    print("An error occurred:", e)

        self._trace = sk_reg_results._trace
        self.SSR = [self._trace[i][1][2] for i in range(1, len(self._trace))]
        OLS_Regimes.__init__(self, y, x_constant, regimes=self.clusters, w=w, name_regimes='Skater_reg', name_y=name_y, name_x=name_x,
                                     constant_regi='many', cols2regi='all', regime_err_sep=True, **kwargs)

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
    from spreg import OLS_Regimes

    db = libpysal.io.open(libpysal.examples.get_path("NAT.dbf"), "r")
    y_var = "HR90"
    y = np.array(db.by_col(y_var)).reshape(-1,1)
    x_var = ['PS90','UE90']
    x = np.array([db.by_col(name) for name in x_var]).T
    r_var = "SOUTH"
    regimes = db.by_col(r_var)
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("NAT.shp"))
    w.transform = "r"
    olsr = OLS_Regimes(
            y,
        x,
        regimes,
        w=w,
        constant_regi="many",
        nonspat_diag=True,
        spat_diag=True,
        name_y=y_var,
        name_x=x_var,
        name_ds="NAT",
        name_regimes=r_var,
        regime_err_sep=True,
        cols2regi=[True, True],
        sig2n_k=False,
        white_test=True,
        #robust="white"
    )
    print(olsr.output)
    print(olsr.summary)
