"""
ML Estimation of Spatial Lag Model with Regimes
"""

__author__ = "Luc Anselin luc.anselin@asu.edu, Pedro V. Amaral pedro.amaral@asu.edu"

import numpy as np
from . import regimes as REGI
from . import user_output as USER
from . import diagnostics as DIAG
import multiprocessing as mp
from .ml_lag import BaseML_Lag
from .utils import set_warn, get_lags
import pandas as pd
from .output import output, _nonspat_top, _spat_diag_out, _spat_pseudo_r2, _summary_impacts


__all__ = ["ML_Lag_Regimes"]


class ML_Lag_Regimes(BaseML_Lag, REGI.Regimes_Frame):

    """
    ML estimation of the spatial lag model with regimes (note no consistency 
    checks, diagnostics or constants added) :cite:`Anselin1988`.

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
    constant_regi: string
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes

                   * 'many': a vector of ones is appended to x and considered different per regime (default)
    cols2regi    : list, 'all'
                   Argument indicating whether each
                   column of x should be considered as different per regime
                   or held constant across regimes (False).
                   If a list, k booleans indicating for each variable the
                   option (True if one per regime, False to be held constant).
                   If 'all' (default), all the variables vary by regime.
    w            : Sparse matrix
                   Spatial weights sparse matrix 
    method       : string
                   if 'full', brute force calculation (full matrix expressions)
                   if 'ord', Ord eigenvalue method
                   if 'LU', LU sparse matrix decomposition
    epsilon      : float
                   tolerance criterion in mimimize_scalar function and inverse_product
    slx_lags     : integer
                   Number of spatial lags of X to include in the model specification.
                   If slx_lags>0, the specification becomes of the Spatial Durbin type.                   
    regime_lag_sep: boolean
                    If True, the spatial parameter for spatial lag is also
                    computed according to different regimes. If False (default),
                    the spatial parameter is fixed accross regimes.
    spat_diag    : boolean
                   If True, then compute Common Factor Hypothesis test when applicable
    spat_impacts : string or list
                   Include average direct impact (ADI), average indirect impact (AII),
                    and average total impact (ATI) in summary results.
                    Options are 'simple', 'full', 'power', 'all' or None.
                    See sputils._spmultiplier for more information.
    cores        : boolean
                   Specifies if multiprocessing is to be used
                   Default: no multiprocessing, cores = False
                   Note: Multiprocessing may not work on all platforms.
    vm           : boolean
                   if True, include variance-covariance matrix in summary
                   results
    name_y       : string
                   Name of dependent variable for use in output
    name_x       : list of strings
                   Names of independent variables for use in output
    name_w       : string
                   Name of weights matrix for use in output
    name_ds      : string
                   Name of dataset for use in output
    name_regimes : string
                   Name of regimes variable for use in output
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
                   (k+1)x1 array of estimated coefficients (rho first)
    rho          : float
                   estimate of spatial autoregressive coefficient
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    u            : array
                   nx1 array of residuals
    predy        : array
                   nx1 array of predicted y values
    n            : integer
                   Number of observations
    k            : integer
                   Number of variables for which coefficients are estimated
                   (including the constant, excluding the rho)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    y            : array
                   nx1 array for dependent variable
    x            : array
                   Two dimensional array with n rows and one column for each
                   independent (exogenous) variable, including the constant
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    method       : string
                   log Jacobian method.
                   if 'full': brute force (full matrix computations)
                   if 'ord', Ord eigenvalue method
                   if 'LU', LU sparse matrix decomposition
    cfh_test     : tuple
                   Common Factor Hypothesis test; tuple contains the pair (statistic,
                   p-value). Only when it applies (see specific documentation).
    epsilon      : float
                   tolerance criterion used in minimize_scalar function and inverse_product
    mean_y       : float
                   Mean of dependent variable
    std_y        : float
                   Standard deviation of dependent variable
    vm           : array
                   Variance covariance matrix (k+1 x k+1), all coefficients
    vm1          : array
                   Variance covariance matrix (k+2 x k+2), includes sig2
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    sig2         : float
                   Sigma squared used in computations
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    logll        : float
                   maximized log-likelihood (including constant terms)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    aic          : float
                   Akaike information criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    schwarz      : float
                   Schwarz criterion
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    predy_e      : array
                   predicted values from reduced form
    e_pred       : array
                   prediction errors using reduced form predicted values
    pr2          : float
                   Pseudo R squared (squared correlation between y and ypred)
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    pr2_e        : float
                   Pseudo R squared (squared correlation between y and ypred_e
                   (using reduced form))
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
    sp_multipliers: dict
                   Dictionary of spatial multipliers (if spat_impacts is not None)
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
                   Name of regimes variable for use in output
    title        : string
                   Name of the regression method used
                   Only available in dictionary 'multi' when multiple regressions
                   (see 'multi' below for details)
    regimes      : list
                   List of n values with the mapping of each
                   observation to a regime. Assumed to be aligned with 'x'.
    constant_regi: ['one', 'many']
                   Ignored if regimes=False. Constant option for regimes.
                   Switcher controlling the constant term setup. It may take
                   the following values:

                   * 'one': a vector of ones is appended to x and held constant across regimes

                   * 'many': a vector of ones is appended to x and considered different per regime
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
                    always set to False - kept for compatibility with other
                    regime models
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

    Open data baltim.dbf using pysal and create the variables matrices and weights matrix.

    >>> import numpy as np
    >>> import libpysal
    >>> from libpysal import examples
    >>> from libpysal.examples import load_example
    >>> from libpysal.weights import Queen
    >>> from spreg import ML_Lag_Regimes
    >>> import geopandas as gpd
    >>> np.set_printoptions(suppress=True) #prevent scientific format
    >>> baltimore = load_example('Baltimore')
    >>> db = libpysal.io.open(baltimore.get_path("baltim.dbf"),'r')
    >>> df = gpd.read_file(baltimore.get_path("baltim.shp"))
    >>> ds_name = "baltim.dbf"
    >>> y_name = "PRICE"
    >>> y = np.array(db.by_col(y_name)).T
    >>> y.shape = (len(y),1)
    >>> x_names = ["NROOM","AGE","SQFT"]
    >>> x = np.array([db.by_col(var) for var in x_names]).T
    >>> w = Queen.from_dataframe(df)
    >>> w_name = "baltim_q.gal"
    >>> w.transform = 'r'    

    Since in this example we are interested in checking whether the results vary
    by regimes, we use CITCOU to define whether the location is in the city or 
    outside the city (in the county):

    >>> regimes = db.by_col("CITCOU")

    Now we can run the regression with all parameters:

    >>> mllag = ML_Lag_Regimes(y,x,regimes,w=w,name_y=y_name,name_x=x_names,\
               name_w=w_name,name_ds=ds_name,name_regimes="CITCOU")
    >>> np.around(mllag.betas, decimals=4)
    array([[-14.5158],
           [  4.4923],
           [ -0.0336],
           [  0.3541],
           [ -3.601 ],
           [  3.8736],
           [ -0.1747],
           [  0.8238],
           [  0.525 ]])
    >>> "{0:.6f}".format(mllag.rho)
    '0.524971'
    >>> "{0:.6f}".format(mllag.mean_y)
    '44.307180'
    >>> "{0:.6f}".format(mllag.std_y)
    '23.606077'
    >>> np.around(np.diag(mllag.vm1), decimals=4)
    array([ 48.6818,   2.4524,   0.0052,   0.0663,  71.4439,   3.2837,
             0.0118,   0.0498,   0.0042, 409.1225])
    >>> np.around(np.diag(mllag.vm), decimals=4)
    array([48.6818,  2.4524,  0.0052,  0.0663, 71.4439,  3.2837,  0.0118,
            0.0498,  0.0042])
    >>> "{0:.6f}".format(mllag.sig2)
    '204.827093'
    >>> "{0:.6f}".format(mllag.logll)
    '-867.086467'
    >>> "{0:.6f}".format(mllag.aic)
    '1752.172934'
    >>> "{0:.6f}".format(mllag.schwarz)
    '1782.339657'
    >>> mllag.title
    'MAXIMUM LIKELIHOOD SPATIAL LAG - REGIMES (METHOD = full)'
    """

    def __init__(
        self,
        y,
        x,
        regimes,
        w=None,
        constant_regi="many",
        cols2regi="all",
        method="full",
        epsilon=0.0000001,
        slx_lags=0,
        regime_lag_sep=False,
        cores=False,
        spat_diag=True,
        spat_impacts="simple",
        vm=False,
        name_y=None,
        name_x=None,
        name_w=None,
        name_ds=None,
        name_regimes=None,
        latex=False,
    ):

        n = USER.check_arrays(y, x)
        y, name_y = USER.check_y(y, n, name_y)
        w = USER.check_weights(w, y, w_required=True, slx_lags=slx_lags)
        name_y = USER.set_name_y(name_y)
        self.name_y = name_y

        x_constant, name_x, warn = USER.check_constant(x, name_x, just_rem=True)
        set_warn(self, warn)
        name_x = USER.set_name_x(name_x, x_constant, constant=True)

        if slx_lags > 0:
            lag_x = get_lags(w, x_constant, slx_lags)
            x_constant = np.hstack((x_constant, lag_x))
            name_x += USER.set_name_spatial_lags(name_x, slx_lags)
            kwx = lag_x.shape[1]

        self.name_x_r = USER.set_name_x(name_x, x_constant) + [USER.set_name_yend_sp(name_y)]
        self.method = method
        self.epsilon = epsilon
        regimes, name_regimes = USER.check_reg_list(regimes, name_regimes, n)
        self.name_regimes = USER.set_name_ds(name_regimes)
        self.constant_regi = constant_regi
        self.n = n
        cols2regi = REGI.check_cols2regi(
            constant_regi, cols2regi, x_constant, add_cons=False
        )
        self.cols2regi = cols2regi
        self.regimes_set = REGI._get_regimes_set(regimes)
        self.regimes = regimes
        self.regime_lag_sep = regime_lag_sep
        self._cache = {}
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        USER.check_regimes(self.regimes_set, self.n, x_constant.shape[1])

        # regime_err_sep is ignored, always False

        if regime_lag_sep == True:
            if not (set(cols2regi) == set([True]) and constant_regi == "many"):
                raise Exception(
                    "All variables must vary by regimes if regime_lag_sep = True."
                )
            cols2regi += [True]
            w_i, regi_ids, warn = REGI.w_regimes(
                w,
                regimes,
                self.regimes_set,
                transform=True,
                get_ids=True,
                min_n=len(cols2regi) + 1,
            )
            set_warn(self, warn)
        else:
            cols2regi += [False]

        if set(cols2regi) == set([True]) and constant_regi == "many":
            self.y = y
            self.ML_Lag_Regimes_Multi(
                y,
                x_constant,
                w_i,
                w,
                regi_ids,
                slx_lags=slx_lags,
                cores=cores,
                cols2regi=cols2regi,
                method=method,
                epsilon=epsilon,
                spat_diag=spat_diag,
                spat_impacts=spat_impacts,
                vm=vm,
                name_y=name_y,
                name_x=name_x,
                name_regimes=self.name_regimes,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        else:
            # if regime_lag_sep == True:
            #    w = REGI.w_regimes_union(w, w_i, self.regimes_set)
            x, self.name_x, x_rlist = REGI.Regimes_Frame.__init__(
                self,
                x_constant,
                regimes,
                constant_regi,
                cols2regi=cols2regi[:-1],
                names=name_x,
                rlist=True
            )
            self.name_x.append("_Global_" + USER.set_name_yend_sp(name_y))
            BaseML_Lag.__init__(self, y=y, x=x, w=w, method=method, epsilon=epsilon)
            self.kf += 1  # Adding a fixed k to account for spatial lag in Chow
            # adding a k to account for spatial lag in aic, sc
            self.k += 1
            self.chow = REGI.Chow(self)
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            self.regime_lag_sep = regime_lag_sep
            self.output = pd.DataFrame(self.name_x, columns=['var_names'])
            self.output['regime'] = x_rlist + ['_Global']
            self.output['var_type'] = ['x'] * (len(self.name_x) - 1) + ['rho']
            self.output['equation'] = 0
            self.slx_lags = slx_lags
            diag_out = None
            if slx_lags > 0:
                self.title = ("MAXIMUM LIKELIHOOD SPATIAL DURBIN - REGIMES"+ " (METHOD = "+ method+ ")")
                fixed_wx = cols2regi[-(kwx+1):-1].count(False)
                kwx = kwx - fixed_wx
                if kwx > 0:
                    for m in self.regimes_set:
                        r_output = self.output[(self.output['regime'] == str(m)) & (self.output['var_type'] == 'x')]
                        wx_index = r_output.index[-kwx:]
                        self.output.loc[wx_index, 'var_type'] = 'wx'
                if fixed_wx > 0:
                    f_wx_index = self.output.index[-(fixed_wx+1):-1]
                    self.output.loc[f_wx_index, 'var_type'] = 'wx'
                if spat_diag and slx_lags == 1:
                    diag_out = _spat_diag_out(self, w, 'yend', ml=True)
            else:
                self.title = ("MAXIMUM LIKELIHOOD SPATIAL LAG - REGIMES"+ " (METHOD = "+ method+ ")")
               
            if spat_impacts:
                self.sp_multipliers, impacts_str = _summary_impacts(self, w, spat_impacts, slx_lags, regimes=True)
                try:
                    diag_out += impacts_str
                except TypeError:
                    diag_out = impacts_str
            self.other_top = _spat_pseudo_r2(self)
            self.other_top += _nonspat_top(self, ml=True)
            output(reg=self, vm=vm, robust=False, other_end=diag_out, latex=latex)

    def ML_Lag_Regimes_Multi(
        self,
        y,
        x,
        w_i,
        w,
        regi_ids,
        slx_lags,
        cores,
        cols2regi,
        method,
        epsilon,
        spat_diag,
        spat_impacts,
        vm,
        name_y,
        name_x,
        name_regimes,
        name_w,
        name_ds,
        latex,
    ):
        #pool = mp.Pool(cores)
        results_p = {}
        """
        for r in self.regimes_set:
            if system() == 'Windows':
                is_win = True
                results_p[r] = _work(*(y,x,regi_ids,r,w_i[r],method,epsilon,name_ds,name_y,name_x,name_w,name_regimes))
            else:                
                results_p[r] = pool.apply_async(_work,args=(y,x,regi_ids,r,w_i[r],method,epsilon,name_ds,name_y,name_x,name_w,name_regimes, ))
                is_win = False
        """
        x_constant, name_x = REGI.check_const_regi(self, x, name_x, regi_ids)
        name_x = name_x + [USER.set_name_yend_sp(name_y)]
        for r in self.regimes_set:
            if cores:
                pool = mp.Pool(None)
                results_p[r] = pool.apply_async(
                    _work,
                    args=(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w_i[r],
                        slx_lags,
                        method,
                        epsilon,
                        name_ds,
                        name_y,
                        name_x,
                        name_w,
                        name_regimes,
                    ),
                )
            else:
                results_p[r] = _work(
                    *(
                        y,
                        x_constant,
                        regi_ids,
                        r,
                        w_i[r],
                        slx_lags,
                        method,
                        epsilon,
                        name_ds,
                        name_y,
                        name_x,
                        name_w,
                        name_regimes,
                    )
                )

        self.kryd = 0
        self.kr = len(cols2regi) + 1
        self.kf = 0
        self.nr = len(self.regimes_set)
        self.name_x_r = name_x
        self.name_regimes = name_regimes
        self.vm = np.zeros((self.nr * self.kr, self.nr * self.kr), float)
        self.betas = np.zeros((self.nr * self.kr, 1), float)
        self.u = np.zeros((self.n, 1), float)
        self.predy = np.zeros((self.n, 1), float)
        self.predy_e = np.zeros((self.n, 1), float)
        self.e_pred = np.zeros((self.n, 1), float)
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
            self.predy_e[
                regi_ids[r],
            ] = results[r].predy_e
            self.e_pred[
                regi_ids[r],
            ] = results[r].e_pred
            self.name_y += results[r].name_y
            self.name_x += results[r].name_x
            results[r].other_top = _spat_pseudo_r2(results[r])
            results[r].other_top += _nonspat_top(results[r], ml=True)
            results[r].other_mid = ""
            if slx_lags > 0:
                kx = (len(results[r].name_x) - 1) // (slx_lags + 1)
                var_types = ['x'] * (kx + 1) + ['wx'] * kx * slx_lags + ['rho']
            else:
                var_types = ['x'] * (len(results[r].name_x) - 1) + ['rho']
            results[r].output = pd.DataFrame({'var_names': results[r].name_x,
                                              'var_type': var_types,
                                              'regime': r, 'equation': r})
            self.output = pd.concat([self.output, results[r].output], ignore_index=True)
            if spat_diag and slx_lags == 1:
                results[r].other_mid += _spat_diag_out(results[r], None, 'yend', ml=True)
            if spat_impacts:
                results[r].sp_multipliers, impacts_str = _summary_impacts(results[r], results[r].w, spat_impacts, slx_lags)
                results[r].other_mid += impacts_str
            counter += 1
        self.multi = results
        self.chow = REGI.Chow(self)
        output(reg=self, vm=vm, robust=False, other_end=False, latex=latex)


def _work(
    y,
    x,
    regi_ids,
    r,
    w_r,
    slx_lags,
    method,
    epsilon,
    name_ds,
    name_y,
    name_x,
    name_w,
    name_regimes,
):
    y_r = y[regi_ids[r]]
    x_r = x[regi_ids[r]]
    model = BaseML_Lag(y_r, x_r, w_r, method=method, epsilon=epsilon)
    if slx_lags == 0:
        model.title = ("MAXIMUM LIKELIHOOD SPATIAL LAG - REGIME "+ str(r)+ " (METHOD = "+ method+ ")")
    else:
        model.title = ("MAXIMUM LIKELIHOOD SPATIAL DURBIN - REGIME "+ str(r)+ " (METHOD = "+ method+ ")")
    model.name_ds = name_ds
    model.name_y = "%s_%s" % (str(r), name_y)
    model.name_x = ["%s_%s" % (str(r), i) for i in name_x]
    model.name_w = name_w
    model.name_regimes = name_regimes
    model.k += 1  # add 1 for proper df and aic, sc
    model.aic = DIAG.akaike(reg=model)
    model.schwarz = DIAG.schwarz(reg=model)
    model.slx_lags = slx_lags
    model.w = w_r
    model.rho = model.betas[-1]
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
    import libpysal as ps

    db = ps.io.open(ps.examples.get_path("baltim.dbf"), "r")
    ds_name = "baltim.dbf"
    y_name = "PRICE"
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y), 1)
    x_names = ["NROOM", "NBATH", "PATIO", "FIREPL", "AC", "GAR", "AGE", "LOTSZ", "SQFT"]
    x = np.array([db.by_col(var) for var in x_names]).T
    ww = ps.io.open(ps.examples.get_path("baltim_q.gal"))
    w = ww.read()
    ww.close()
    w_name = "baltim_q.gal"
    w.transform = "r"
    regimes = db.by_col("CITCOU")

    mllag = ML_Lag_Regimes(
        y,
        x,
        regimes,
        w=w,
        method="full",
        name_y=y_name,
        name_x=x_names,
        name_w=w_name,
        name_ds=ds_name,
        regime_lag_sep=True,
        regime_err_sep=False,
        constant_regi="many",
        name_regimes="CITCOU",
    )
    print(mllag.output)
    print(mllag.summary)
