import numpy as np
import multiprocessing as mp
import pandas as pd
from . import regimes as REGI
from . import user_output as USER
from .utils import set_warn, RegressionProps_basic, spdot, sphstack, get_lags, optim_k
from .twosls import BaseTSLS
from .robust import hac_multi
from .output import output, _spat_diag_out
from .skater_reg import Skater_reg

"""
Two-stage Least Squares estimation with regimes.
"""

__author__ = "Luc Anselin, Pedro V. Amaral, David C. Folch"


class TSLS_Regimes(BaseTSLS, REGI.Regimes_Frame):

    """
    Two stage least squares (2SLS) with regimes.

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
    robust       : string
                   If 'white', then a White consistent estimator of the
                   variance-covariance matrix is given.
                   If 'hac', then a HAC consistent estimator of the
                   variance-covariance matrix is given.
                   If 'ogmm', then Optimal GMM is used to estimate
                   betas and the variance-covariance matrix.
                   Default set to None.
    gwk          : pysal W object
                   Kernel spatial weights needed for HAC estimation. Note:
                   matrix must have ones along the main diagonal.
    slx_lags       : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the SLX type.
                   Note: WX is computed using the complete weights matrix
    slx_vars     : either "all" (default) or list of booleans to select x variables
                   to be lagged                         
    sig2n_k      : boolean
                   If True, then use n-k to estimate sigma^2. If False, use n.
    vm           : boolean
                   If True, include variance-covariance matrix in summary
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
    name_regimes : string
                   Name of regimes variable for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    vm           : array
                   Variance covariance matrix (kxk)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: [False, 'one', 'many']
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime (default).
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
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
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
    >>> from libpysal.weights import Rook
    >>> import geopandas as gpd

    Open data on NCOVR US County Homicides (3085 areas) using geopandas.

    >>> nat = load_example('Natregimes')
    >>> db = gpd.read_file(nat.get_path('natregimes.shp'))

    Select the HR90 column (homicide rates in 1990) as the
    dependent variable for the regression. 

    >>> y = db['HR90']

    Select UE90 (unemployment rate) and PS90 (population structure) variables
    as independent variables in the regression.

    >>> x = db[['PS90','UE90']]

    In this case we consider RD90 (resource deprivation) as an endogenous regressor.
    We tell the model that this is so by passing it in a different parameter
    from the exogenous variables (x).

    >>> yd = db['RD90']

    Because we have endogenous variables, to obtain a correct estimate of the
    model, we need to instrument for RD90. We use FP89 (families below poverty)
    for this and hence put it in the instruments parameter, 'q'.

    >>> q = db[['FP89']]

    The different regimes in this data are given according to the North and
    South dummy (SOUTH).

    >>> r_var = db['SOUTH']

    Since we want to perform tests for spatial dependence, we need to specify
    the spatial weights matrix that includes the spatial configuration of the
    observations into the error component of the model. To do that, we can open
    an already existing gal file or create a new one. In this case, we will
    create one from our geopandas dataframe.

    >>> w = Rook.from_dataframe(db, use_index=True)

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We can now run the regression and then have a summary of the output
    by typing: model.summary
    Alternatively, we can just check the betas and standard errors of the
    parameters:

    >>> from spreg import TSLS_Regimes
    >>> tslsr = TSLS_Regimes(y, x, yd, q, r_var, w=w, spat_diag=True, name_regimes='SOUTH', name_ds='NAT', name_w='NAT.shp')

    >>> tslsr.betas
    array([[ 3.66973562],
           [ 1.06950466],
           [ 0.14680946],
           [ 9.55873243],
           [ 1.94666348],
           [-0.30810214],
           [ 2.45864196],
           [ 3.68718119]])

    >>> np.sqrt(tslsr.vm.diagonal())
    array([0.46522151, 0.12074672, 0.05661795, 0.41893265, 0.16721773,
           0.06631022, 0.27538921, 0.21745974])

    >>> print(tslsr.summary)
    REGRESSION RESULTS
    ------------------
    <BLANKLINE>
    SUMMARY OF OUTPUT: TWO STAGE LEAST SQUARES - REGIMES
    ----------------------------------------------------
    Data set            :         NAT
    Weights matrix      :     NAT.shp
    Dependent Variable  :        HR90                Number of Observations:        3085
    Mean dependent var  :      6.1829                Number of Variables   :           8
    S.D. dependent var  :      6.6414                Degrees of Freedom    :        3077
    Pseudo R-squared    :      0.4246
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     z-Statistic     Probability
    ------------------------------------------------------------------------------------
              0_CONSTANT         3.66974         0.46522         7.88815         0.00000
                  0_PS90         1.06950         0.12075         8.85742         0.00000
                  0_UE90         0.14681         0.05662         2.59298         0.00951
                  0_RD90         2.45864         0.27539         8.92788         0.00000
              1_CONSTANT         9.55873         0.41893        22.81687         0.00000
                  1_PS90         1.94666         0.16722        11.64149         0.00000
                  1_UE90        -0.30810         0.06631        -4.64637         0.00000
                  1_RD90         3.68718         0.21746        16.95570         0.00000
    ------------------------------------------------------------------------------------
    Instrumented: 0_RD90, 1_RD90
    Instruments: 0_FP89, 1_FP89
    Regimes variable: SOUTH
    <BLANKLINE>
    REGIMES DIAGNOSTICS - CHOW TEST
                     VARIABLE        DF        VALUE           PROB
                CONSTANT              1         88.485           0.0000
                    PS90              1         18.086           0.0000
                    UE90              1         27.220           0.0000
                    RD90              1         12.258           0.0005
             Global test              4        195.994           0.0000
    <BLANKLINE>
    DIAGNOSTICS FOR SPATIAL DEPENDENCE
    TEST                              DF         VALUE           PROB
    Anselin-Kelejian Test             1        104.255           0.0000
    ================================ END OF REPORT =====================================
    """

    def __init__(
        self,
        y,
        x,
        yend,
        q,
        regimes,
        w=None,
        robust=None,
        gwk=None,
        slx_lags=0,
        slx_vars="all",
        sig2n_k=True,
        spat_diag=False,
        vm=False,
        constant_regi="many",
        cols2regi="all",
        regime_err_sep=False,
        name_y=None,
        name_x=None,
        cores=False,
        name_yend=None,
        name_q=None,
        name_regimes=None,
        name_w=None,
        name_gwk=None,
        name_ds=None,
        summ=True,
        latex=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        USER.check_robust(robust, gwk)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        if robust == "hac":
            if regime_err_sep:
                set_warn(
                    self,
                    "Error by regimes is not available for HAC estimation. The error by regimes has been disabled for this model.",
                )
                regime_err_sep = False
        spat_diag, warn = USER.check_spat_diag(spat_diag=spat_diag, w=w, robust=robust, slx_lags=slx_lags)
        set_warn(self, warn)
        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)
        w = USER.check_weights(w, y, slx_lags=slx_lags, w_required=spat_diag)
        if slx_lags > 0:
            x_constant,name_x = USER.flex_wx(w,x=x_constant,name_x=name_x,constant=False,
                                                slx_lags=slx_lags,slx_vars=slx_vars)
            set_warn(self,"WX is computed using the complete W, i.e. not trimmed by regimes.")

        self.constant_regi = constant_regi
        self.cols2regi = cols2regi
        self.name_ds = USER.set_name_ds(name_ds)
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.name_w = USER.set_name_w(name_w, w)
        self.name_gwk = USER.set_name_w(name_gwk, gwk)
        self.name_y = USER.set_name_y(name_y)
        name_yend = USER.set_name_yend(name_yend, yend)
        name_q = USER.set_name_q(name_q, q)
        self.name_x_r = USER.set_name_x(name_x, x_constant) + name_yend
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x_constant, yend=yend, add_cons=False
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
            self._tsls_regimes_multi(
                x_constant,
                yend,
                q,
                w,
                regi_ids,
                cores,
                gwk,
                slx_lags,
                sig2n_k,
                robust,
                spat_diag,
                vm,
                name_x,
                name_yend,
                name_q,
                summ,
                latex
            )
        else:
            q, self.name_q, xtype = REGI.Regimes_Frame.__init__(
                self, q, regimes, constant_regi=None, cols2regi="all", names=name_q
            )
            x, self.name_x, xtype, x_rlist = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes,
                constant_regi,
                cols2regi=cols2regi,
                names=name_x,
                rlist=True
            )
            yend, self.name_yend, xtypeyd, yend_rlist = REGI.Regimes_Frame.__init__(
                self,
                yend,
                regimes,
                constant_regi=None,
                cols2regi=cols2regi,
                yend=True,
                names=name_yend,
                rlist=True
            )
            self.output = pd.DataFrame(self.name_x+self.name_yend,
                                       columns=['var_names'])
            self.output['var_type'] = xtype+xtypeyd
            self.output['regime'] = x_rlist+yend_rlist
            self.output['equation'] = 0

            BaseTSLS.__init__(
                self, y=y, x=x, yend=yend, q=q, robust=robust, gwk=gwk, sig2n_k=sig2n_k
            )

            if slx_lags == 0:
                self.title = "TWO STAGE LEAST SQUARES - REGIMES"
            else:
                self.title = "TWO STAGE LEAST SQUARES WITH SPATIALLY LAGGED X (2SLS-SLX) - REGIMES"

            if robust == "ogmm":
                _optimal_weight(self, sig2n_k)
            self.name_z = self.name_x + self.name_yend
            self.name_h = USER.set_name_h(self.name_x, self.name_q)
            self.chow = REGI.Chow(self)
            self.robust = USER.set_robust(robust)
            if summ:
                if spat_diag:
                    diag_out = _spat_diag_out(self, w, 'yend')
                else:
                    diag_out = None
                output(reg=self, vm=vm, robust=robust, other_end=diag_out, latex=latex)


    def _tsls_regimes_multi(
        self,
        x,
        yend,
        q,
        w,
        regi_ids,
        cores,
        gwk,
        slx_lags,
        sig2n_k,
        robust,
        spat_diag,
        vm,
        name_x,
        name_yend,
        name_q,
        summ,
        latex
    ):
        results_p = {}
        """
        for r in self.regimes_set:
            if system() != 'Windows':
                is_win = True
                results_p[r] = _work(*(self.y,x,w,regi_ids,r,yend,q,robust,sig2n_k,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes))
            else:
                pool = mp.Pool(cores)
                results_p[r] = pool.apply_async(_work,args=(self.y,x,w,regi_ids,r,yend,q,robust,sig2n_k,self.name_ds,self.name_y,name_x,name_yend,name_q,self.name_w,self.name_regimes))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        self.name_x_r = name_x + name_yend
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
                        yend,
                        q,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        name_yend,
                        name_q,
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
                        yend,
                        q,
                        robust,
                        sig2n_k,
                        self.name_ds,
                        self.name_y,
                        name_x,
                        name_yend,
                        name_q,
                        self.name_w,
                        self.name_regimes,
                        slx_lags
                    )
                )

        self.kryd = 0
        self.kr = x_constant.shape[1] + yend.shape[1]
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
        (
            self.name_y,
            self.name_x,
            self.name_yend,
            self.name_q,
            self.name_z,
            self.name_h,
        ) = ([], [], [], [], [], [])
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
                (counter * self.kr) : ((counter + 1) * self.kr),
                (counter * self.kr) : ((counter + 1) * self.kr),
            ] = results[r].vm
            self.betas[
                (counter * self.kr) : ((counter + 1) * self.kr),
            ] = results[r].betas
            self.u[
                regi_ids[r],
            ] = results[r].u
            self.predy[
                regi_ids[r],
            ] = results[r].predy
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            self.name_yend += results[r].name_yend
            self.name_q += results[r].name_q
            self.name_z += results[r].name_z
            self.name_h += results[r].name_h
            self.output = pd.concat([self.output, pd.DataFrame({'var_names': results[r].name_x+results[r].name_yend,
                                                       'var_type': ['o']+['x']*(len(results[r].name_x)-1)+['yend']*len(results[r].name_yend),
                                                       'regime': r, 'equation': r})], ignore_index=True)

            counter += 1
        self.multi = results

        self.hac_var = sphstack(x_constant[:, 1:], q)
        if robust == "hac":
            hac_multi(self, gwk)
        if robust == "ogmm":
            set_warn(
                self,
                "Residuals treated as homoskedastic for the purpose of diagnostics.",
            )
        self.chow = REGI.Chow(self)
        if spat_diag:
            self._get_spat_diag_props(results, regi_ids, x_constant, yend, q)
            diag_out = _spat_diag_out(self, w, 'yend')
        else:
            diag_out = None
        if summ:
            self.output.sort_values(by='regime', inplace=True)
            output(reg=self, vm=vm, robust=robust, other_end=diag_out, latex=latex)

    def _get_spat_diag_props(self, results, regi_ids, x, yend, q):
        self._cache = {}
        x = REGI.regimeX_setup(x, self.regimes, [True] * x.shape[1], self.regimes_set)
        self.z = sphstack(
            x,
            REGI.regimeX_setup(
                yend, self.regimes, [True] * yend.shape[1], self.regimes_set
            ),
        )
        self.h = sphstack(
            x,
            REGI.regimeX_setup(q, self.regimes, [True] * q.shape[1], self.regimes_set),
        )
        hthi = np.linalg.inv(spdot(self.h.T, self.h))
        zth = spdot(self.z.T, self.h)
        self.varb = np.linalg.inv(spdot(spdot(zth, hthi), zth.T))


def _work(
    y,
    x,
    w,
    regi_ids,
    r,
    yend,
    q,
    robust,
    sig2n_k,
    name_ds,
    name_y,
    name_x,
    name_yend,
    name_q,
    name_w,
    name_regimes,
    slx_lags,
):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    yend_r = yend[regi_ids[r]]
    q_r = q[regi_ids[r]]
    if robust == "hac" or robust == "ogmm":
        robust2 = None
    else:
        robust2 = robust
    model = BaseTSLS(y_r, x_r, yend_r, q_r, robust=robust2, sig2n_k=sig2n_k)
    if slx_lags == 0:
        model.title = "TWO STAGE LEAST SQUARES ESTIMATION - REGIME %s" % r
    else:
        model.title = "TWO STAGE LEAST SQUARES ESTIMATION WITH SLX - REGIME %s" % r
    if robust == "ogmm":
        _optimal_weight(model, sig2n_k, warn=False)
    model.robust = USER.set_robust(robust)
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_yend = ["%s_%s" % (str(r), i) for i in name_yend]
    model.name_z = model.name_x + model.name_yend
    model.name_q = ["%s_%s" % (str(r), i) for i in name_q]
    model.name_h = model.name_x + model.name_q
    model.name_w = name_w
    model.name_regimes = name_regimes
    if w:
        w_r, warn = REGI.w_regime(w, regi_ids[r], r, transform=True)
        set_warn(model, warn)
        model.w = w_r
    return model


def _optimal_weight(reg, sig2n_k, warn=True):
    try:
        Hu = reg.h.toarray() * reg.u ** 2
    except:
        Hu = reg.h * reg.u ** 2
    if sig2n_k:
        S = spdot(reg.h.T, Hu, array_out=True) / (reg.n - reg.k)
    else:
        S = spdot(reg.h.T, Hu, array_out=True) / reg.n
    Si = np.linalg.inv(S)
    ZtH = spdot(reg.z.T, reg.h)
    ZtHSi = spdot(ZtH, Si)
    fac2 = np.linalg.inv(spdot(ZtHSi, ZtH.T, array_out=True))
    fac3 = spdot(ZtHSi, spdot(reg.h.T, reg.y), array_out=True)
    betas = np.dot(fac2, fac3)
    if sig2n_k:
        vm = fac2 * (reg.n - reg.k)
    else:
        vm = fac2 * reg.n
    RegressionProps_basic(reg, betas=betas, vm=vm, sig2=False)
    #reg.title += " (Optimal-Weighted GMM)"
    if warn:
        set_warn(
            reg, "Residuals treated as homoskedastic for the purpose of diagnostics."
        )
    return

class TSLS_Endog_Regimes(TSLS_Regimes):

    """
    Two stage least squares (S2SLS) with endogenous regimes. 
    Based on the function skater_reg as shown in :cite:`Anselin2021`.

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
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_q       : list of strings
                   Names of instruments for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
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
    kstar        : integer
                   Number of endogenous variables.
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
    q            : array
                   Two dimensional array with n rows and one column for each
                   external exogenous variable used as instruments
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    z            : array
                   nxk array of variables (combination of x and yend)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    h            : array
                   nxl array of instruments (combination of x and q)
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
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    utu          : float
                   Sum of squared residuals
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
    ak_test      : tuple
                   Anselin-Kelejian test; tuple contains the pair (statistic,
                   p-value)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    cfh_test     : tuple
                   Common Factor Hypothesis test; tuple contains the pair (statistic,
                   p-value). Only when it applies (see specific documentation).
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_yend    : list of strings
                   Names of endogenous variables for use in output
    name_z       : list of strings
                   Names of exogenous and endogenous variables for use in
                   output
    name_q       : list of strings
                   Names of external instruments
    name_h       : list of strings
                   Names of all instruments used in ouput
    name_w       : string
                   Name of weights matrix for use in output
    name_gwk     : string
                   Name of kernel weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2n        : float
                   Sigma squared (computed with n in the denominator)
    sig2n_k      : float
                   Sigma squared (computed with n-k in the denominator)
    hth          : float
                   :math:`H'H`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    hthi         : float
                   :math:`(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    varb         : array
                   :math:`(Z'H (H'H)^{-1} H'Z)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    zthhthi      : array
                   :math:`Z'H(H'H)^{-1}`.
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pfora1a2     : array
                   n(zthhthi)'varb
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: string
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes.

                   * 'many': a vector of ones is appended to x and considered different per regime.
    cols2regi    : list, 'all'
                   Ignored if regimes=False. Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all', all the variables vary by regime.
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
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
    >>> from spreg import TSLS_Endog_Regimes

    Open data on Baltimore house sales price and characteristics in Baltimore
    from libpysal examples using geopandas.

    >>> db = gpd.read_file(libpysal.examples.get_path('baltim.shp'))

    We will create a weights matrix based on contiguity.

    >>> w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
    >>> w.transform = "r"

    For this example, just to show how the function works, we will use 
    the 'PRICE' column as the dependent variable, 'AGE' as independent variable, 
    'SQFT' as endogenous variable, and 'NROOM' as an instrument.
    At this point, we will let the model choose the number of clusters.

    >>> reg = TSLS_Endog_Regimes(y=db['PRICE'], x=db['AGE'], yend=db['SQFT'], q=db['NROOM'], w=w)
    
    The function `print(reg.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes:
    >>> reg.betas
    array([[28.55593008],
           [-0.27351083],
           [22.36160147],
           [-0.64768173],
           [22.89318454],
           [-0.48438473],
           [ 1.11252432],
           [ 2.74757415],
           [ 2.26013585]])
    >>> reg.SSR
    [77331.25758888898, 74978.74262486391, 73326.08720216743]
    >>> reg.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 2,
           2, 2, 2, 2, 2, 2, 0, 2, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0,
           1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 2, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int32)

    We  will now set the number of clusters to 2 and run the regression again.

    >>> reg = TSLS_Endog_Regimes(y=db['PRICE'], x=db['AGE'], yend=db['SQFT'], q=db['NROOM'], w=w, n_clusters=2)

    The function `print(reg.summary)` can be used to visualize the results of the regression.

    Alternatively, we can check individual attributes as before:
    >>> reg.betas
    array([[29.89584104],
           [-0.36936638],
           [22.36160147],
           [-0.64768173],
           [ 1.41273696],
           [ 2.74757415]])
    >>> reg.SSR
    [77331.25758888898]
    >>> reg.clusters
    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0,
           0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
           1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int32)
    """


    def __init__(
        self, y, x, yend, q, w=None, n_clusters=None, quorum=-1, trace=True, name_y=None, name_x=None, name_yend=None, name_q=None, 
        constant_regi='many', cols2regi='all', regime_err_sep=True, **kwargs):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        yend, q, name_yend, name_q = USER.check_endog([yend, q], [name_yend, name_q])
        w = USER.check_weights(w, y, w_required=True)
        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        warn = USER.check_regi_args(constant_regi, cols2regi, regime_err_sep)
        set_warn(self, warn)
        # Standardize the variables
        x_combined = np.hstack((x_constant, yend))
        x_std = (x_combined - np.mean(x_combined, axis=0)) / np.std(x_combined, axis=0)

        if quorum < 0:
            quorum = np.max([(x_constant.shape[1]+1)*10, 30])
        
        if not n_clusters:
            n_clusters_opt = x_constant.shape[0]*0.70//quorum
            if n_clusters_opt < 2:
                raise ValueError(
                    "The combination of the values of `N` and `quorum` is not compatible with regimes estimation.")
            sk_reg_results = Skater_reg().fit(n_clusters_opt, w, x_std, {'reg':BaseTSLS,'y':y,'x':x_constant,'yend':yend,'q':q}, quorum=quorum, trace=True)
            n_clusters = optim_k([sk_reg_results._trace[i][1][2] for i in range(1, len(sk_reg_results._trace))])
            self.clusters = sk_reg_results._trace[n_clusters-1][0]
            self.score = sk_reg_results._trace[n_clusters-1][1][2]
        else:
            try:
                sk_reg_results = Skater_reg().fit(n_clusters, w, x_std, {'reg':BaseTSLS,'y':y,'x':x_constant,'yend':yend,'q':q}, quorum=quorum, trace=trace)
                self.clusters = sk_reg_results.current_labels_
                self.score = sk_reg_results._trace[-1][1][2]
            except Exception as e:
                if str(e) == "one or more input arrays have more columns than rows":
                    raise ValueError("One or more input ended up with more variables than observations. Please check your setting for `quorum`.")
                else:
                    print("An error occurred:", e)

        self._trace = sk_reg_results._trace
        self.SSR = [self._trace[i][1][2] for i in range(1, len(self._trace))]
        TSLS_Regimes.__init__(self, y, x_constant, yend, q, regimes=self.clusters, w=w, name_y=name_y, name_x=name_x, name_yend=name_yend, 
                              name_q=name_q, name_regimes='Skater_reg', constant_regi='many', cols2regi='all', regime_err_sep=True, **kwargs)


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
    from libpysal.examples import load_example

    nat = load_example("Natregimes")
    db = libpysal.io.open(nat.get_path("natregimes.dbf"), "r")
    y_var = "HR60"
    y = np.array([db.by_col(y_var)]).T
    x_var = ["PS60", "DV60", "RD60"]
    x = np.array([db.by_col(name) for name in x_var]).T
    yd_var = ["UE60"]
    yd = np.array([db.by_col(name) for name in yd_var]).T
    q_var = ["FP59", "MA60"]
    q = np.array([db.by_col(name) for name in q_var]).T
    r_var = "SOUTH"
    regimes = db.by_col(r_var)
    w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))
    w.transform = "r"
    tslsr = TSLS_Regimes(
        y,
        x,
        yd,
        q,
        regimes,
        w = w,
        constant_regi="many",
        spat_diag=True,
        name_y=y_var,
        name_x=x_var,
        name_yend=yd_var,
        name_q=q_var,
        name_regimes=r_var,
        #cols2regi=[False, True, True, False],
        sig2n_k=False,
        regime_err_sep = True,
        #robust = 'hac',
        vm = False
    )
    print(tslsr.output)
    print(tslsr.summary)




