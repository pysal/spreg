"""
Spatial Lag SUR estimation
"""

__author__ = "Luc Anselin lanselin@gmail.com,    \
             Pedro V. Amaral pedrovma@gmail.com"


import numpy as np
from . import summary_output as SUMMARY
from . import user_output as USER
from . import regimes as REGI
from .sur import BaseThreeSLS
from .diagnostics_sur import sur_setp, sur_chow, sur_joinrho
from .sur_utils import check_k

__all__ = ["SURlagIV"]


class SURlagIV(BaseThreeSLS, REGI.Regimes_Frame):
    """
    User class for spatial lag estimation using IV

    Parameters
    ----------
    bigy       : dictionary
                 with vector for dependent variable by equation
    bigX       : dictionary
                 with matrix of explanatory variables by equation
                 (note, already includes constant term)
    bigyend    : dictionary
                 with matrix of endogenous variables by equation
                 (optional)
    bigq       : dictionary
                 with matrix of instruments by equation
                 (optional)
    w          : spatial weights object, required
    vm         : boolean
                 listing of full variance-covariance matrix, default = False
    w_lags     : integer
                 order of spatial lags for WX instruments, default = 1
    lag_q      : boolean
                 flag to apply spatial lag to other instruments,
                 default = True
    nonspat_diag : boolean
                   flag for non-spatial diagnostics, default = True
    spat_diag    : boolean
                   flag for spatial diagnostics, default = False
    name_bigy  : dictionary
                 with name of dependent variable for each equation.
                 default = None, but should be specified.
                 is done when sur_stackxy is used.
    name_bigX  : dictionary
                 with names of explanatory variables for each
                 equation.
                 default = None, but should be specified.
                 is done when sur_stackxy is used.
    name_bigyend : dictionary
                   with names of endogenous variables for each
                   equation.
                   default = None, but should be specified.
                   is done when sur_stackZ is used.
    name_bigq  : dictionary
                 with names of instrumental variables for each
                 equations.
                 default = None, but should be specified.
                 is done when sur_stackZ is used.
    name_ds    : string
                 name for the data set
    name_w     : string
                 name for the spatial weights

    Attributes
    ----------
    w           : spatial weights object
    bigy        : dictionary
                  with y values
    bigZ        : dictionary
                  with matrix of exogenous and endogenous variables
                  for each equation
    bigyend     : dictionary
                  with matrix of endogenous variables for each
                  equation; contains Wy only if no other endogenous specified
    bigq        : dictionary
                  with matrix of instrumental variables for each
                  equation; contains WX only if no other endogenous specified
    bigZHZH     : dictionary
                  with matrix of cross products Zhat_r'Zhat_s
    bigZHy      : dictionary
                  with matrix of cross products Zhat_r'y_end_s
    n_eq        : int
                  number of equations
    n           : int
                  number of observations in each cross-section
    bigK        : array
                  vector with number of explanatory variables (including constant,
                  exogenous and endogenous) for each equation
    b2SLS       : dictionary
                  with 2SLS regression coefficients for each equation
    tslsE       : array
                  N x n_eq array with OLS residuals for each equation
    b3SLS       : dictionary
                  with 3SLS regression coefficients for each equation
    varb        : array
                  variance-covariance matrix
    sig         : array
                  Sigma matrix of inter-equation error covariances
    resids      : array
                  n by n_eq array of residuals
    corr        : array
                  inter-equation 3SLS error correlation matrix
    tsls_inf    : dictionary
                  with standard error, asymptotic t and p-value,
                  one for each equation
    joinrho     : tuple
                  test on joint significance of spatial autoregressive coefficient.
                  tuple with test statistic, degrees of freedom, p-value
    surchow     : array
                  list with tuples for Chow test on regression coefficients
                  each tuple contains test value, degrees of freedom, p-value
    name_w     : string
                 name for the spatial weights
    name_ds    : string
                 name for the data set
    name_bigy  : dictionary
                 with name of dependent variable for each equation
    name_bigX  : dictionary
                 with names of explanatory variables for each
                 equation
    name_bigyend : dictionary
                   with names of endogenous variables for each
                   equation
    name_bigq  : dictionary
                 with names of instrumental variables for each
                 equations


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
    Each equation should be listed separately. Although not required,
    in this example we will specify additional endogenous regressors.
    Equation 1 has HR80 as dependent variable, PS80 and UE80 as exogenous regressors,
    RD80 as endogenous regressor and FP79 as additional instrument.
    For equation 2, HR90 is the dependent variable, PS90 and UE90 the
    exogenous regressors, RD90 as endogenous regressor and FP99 as
    additional instrument

    >>> y_var = ['HR80','HR90']
    >>> x_var = [['PS80','UE80'],['PS90','UE90']]
    >>> yend_var = [['RD80'],['RD90']]
    >>> q_var = [['FP79'],['FP89']]

    The SUR method requires data to be provided as dictionaries. PySAL
    provides two tools to create these dictionaries from the list of variables:
    sur_dictxy and sur_dictZ. The tool sur_dictxy can be used to create the
    dictionaries for Y and X, and sur_dictZ for endogenous variables (yend) and
    additional instruments (q).

    >>> bigy,bigX,bigyvars,bigXvars = spreg.sur_dictxy(db,y_var,x_var)
    >>> bigyend,bigyendvars = spreg.sur_dictZ(db,yend_var)
    >>> bigq,bigqvars = spreg.sur_dictZ(db,q_var)

    To run a spatial lag model, we need to specify the spatial weights matrix.
    To do that, we can open an already existing gal file or create a new one.
    In this example, we will create a new one from NAT.shp and transform it to
    row-standardized.

    >>> w = Queen.from_shapefile(nat.get_path("natregimes.shp"))
    >>> w.transform='r'

    We can now run the regression and then have a summary of the output by typing:
    print(reg.summary)

    Alternatively, we can just check the betas and standard errors, asymptotic t
    and p-value of the parameters:

    >>> reg = spreg.SURlagIV(bigy,bigX,bigyend,bigq,w=w,name_bigy=bigyvars,name_bigX=bigXvars,name_bigyend=bigyendvars,name_bigq=bigqvars,name_ds="NAT",name_w="nat_queen")
    >>> reg.b3SLS
    {0: array([[ 6.95472387],
           [ 1.44044301],
           [-0.00771893],
           [ 3.65051153],
           [ 0.00362663]]), 1: array([[ 5.61101925],
           [ 1.38716801],
           [-0.15512029],
           [ 3.1884457 ],
           [ 0.25832185]])}

    >>> reg.tsls_inf
    {0: array([[ 0.49128435, 14.15620899,  0.        ],
           [ 0.11516292, 12.50787151,  0.        ],
           [ 0.03204088, -0.2409087 ,  0.80962588],
           [ 0.1876025 , 19.45875745,  0.        ],
           [ 0.05450628,  0.06653605,  0.94695106]]), 1: array([[ 0.44969956, 12.47726211,  0.        ],
           [ 0.10440241, 13.28674277,  0.        ],
           [ 0.04150243, -3.73761961,  0.00018577],
           [ 0.19133145, 16.66451427,  0.        ],
           [ 0.04394024,  5.87893596,  0.        ]])}
    """

    def __init__(
        self,
        bigy,
        bigX,
        bigyend=None,
        bigq=None,
        w=None,
        regimes=None,
        vm=False,
        regime_lag_sep=False,
        w_lags=1,
        lag_q=True,
        nonspat_diag=True,
        spat_diag=False,
        name_bigy=None,
        name_bigX=None,
        name_bigyend=None,
        name_bigq=None,
        name_ds=None,
        name_w=None,
        name_regimes=None,
    ):
        if w is None:
            raise Exception("Spatial weights required for SUR-Lag")
        self.w = w
        WS = w.sparse
        self.name_ds = USER.set_name_ds(name_ds)
        self.name_w = USER.set_name_w(name_w, w)
        if bigyend and not (bigq):
            raise Exception("Instruments needed when endogenous variables")
        # initialize
        self.bigy = bigy
        self.n_eq = len(self.bigy.keys())
        if name_bigy:
            self.name_bigy = name_bigy
        else:  # need to construct y names
            self.name_bigy = {}
            for r in range(self.n_eq):
                yn = "dep_var_" + str(r + 1)
                self.name_bigy[r] = yn
        #        self.bigX = bigX
        if name_bigX is None:
            name_bigX = {}
            for r in range(self.n_eq):
                k = bigX[r].shape[1] - 1
                name_x = ["var_" + str(i + 1) + "_" + str(r + 1) for i in range(k)]
                ct = "Constant_" + str(r + 1)  # NOTE: constant always included in X
                name_x.insert(0, ct)
                name_bigX[r] = name_x
        if name_bigyend is None:
            name_bigyend = {}
        if bigyend is not None:  # check on other endogenous
            self.bigyend = bigyend
            for r in range(self.n_eq):
                ky = bigyend[r].shape[1]
                name_ye = ["end_" + str(i + 1) + "_" + str(r + 1) for i in range(ky)]
                name_bigyend[r] = name_ye
        if name_bigq is None:
            name_bigq = {}
        if bigq is not None:  # check on instruments
            self.bigq = bigq
            for r in range(self.n_eq):
                ki = bigq[r].shape[1]
                name_i = ["inst_" + str(i + 1) + "_" + str(r + 1) for i in range(ki)]
                name_bigq[r] = name_i

        if regimes is not None:
            self.constant_regi = "many"
            self.cols2regi = "all"
            self.regime_err_sep = False
            self.name_regimes = USER.set_name_ds(name_regimes)
            self.regimes_set = REGI._get_regimes_set(regimes)
            self.regimes = regimes
            cols2regi_dic = {}
            self.name_bigX, self.name_bigq, self.name_bigyend = {}, {}, {}
            self.name_x_r = name_bigX

            # spatial lag dependent variable varying across regimes
            if regime_lag_sep == True:
                bigyend, name_bigyend = _get_spatial_lag(
                    self, bigyend, WS, name_bigyend
                )

            for r in range(self.n_eq):
                if bigyend is not None:
                    self.name_x_r[r] += name_bigyend[r]
                    cols2regi_dic[r] = REGI.check_cols2regi(
                        self.constant_regi,
                        self.cols2regi,
                        bigX[r],
                        yend=bigyend[r],
                        add_cons=False,
                    )
                else:
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
                if bigq is not None:
                    bigq[r], self.name_bigq[r] = REGI.Regimes_Frame.__init__(
                        self,
                        bigq[r],
                        regimes,
                        constant_regi=None,
                        cols2regi="all",
                        names=name_bigq[r],
                    )
                if bigyend is not None:
                    bigyend[r], self.name_bigyend[r] = REGI.Regimes_Frame.__init__(
                        self,
                        bigyend[r],
                        regimes,
                        constant_regi=None,
                        cols2regi=cols2regi_dic[r],
                        yend=True,
                        names=name_bigyend[r],
                    )
        else:
            self.name_bigX, self.name_bigq, self.name_bigyend = (
                name_bigX,
                name_bigq,
                name_bigyend,
            )

        # spatial lag dependent variable fixed across regimes or no regimes
        if regimes is None or regime_lag_sep == False:
            bigyend, self.name_bigyend = _get_spatial_lag(
                self, bigyend, WS, name_bigyend
            )
        # spatially lagged exogenous variables
        bigwx = {}
        wxnames = {}
        if w_lags == 1:
            for r in range(self.n_eq):
                bigwx[r] = WS * bigX[r][:, 1:]
                wxnames[r] = ["W_" + i for i in self.name_bigX[r][1:]]
            if bigq:  # other instruments
                if lag_q:  # also lags for instruments
                    bigwq = {}
                    for r in range(self.n_eq):
                        bigwq = WS * bigq[r]
                        bigq[r] = np.hstack((bigq[r], bigwx[r], bigwq))
                        wqnames = ["W_" + i for i in self.name_bigq[r]]
                        wxnames[r] = wxnames[r] + wqnames
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
                else:  # no lags for other instruments
                    for r in range(self.n_eq):
                        bigq[r] = np.hstack((bigq[r], bigwx[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
            else:  # no other instruments only wx
                bigq = {}
                for r in range(self.n_eq):
                    bigq[r] = bigwx[r]
                    self.name_bigq[r] = wxnames[r]
        elif w_lags > 1:  # higher order lags for WX
            for r in range(self.n_eq):
                bigwxwork = WS * bigX[r][:, 1:]
                bigwx[r] = bigwxwork
                nameswork = ["W_" + i for i in self.name_bigX[r][1:]]
                wxnames[r] = nameswork
                for i in range(1, w_lags):
                    bigwxwork = WS * bigwxwork
                    bigwx[r] = np.hstack((bigwx[r], bigwxwork))
                    nameswork = ["W" + i for i in nameswork]
                    wxnames[r] = wxnames[r] + nameswork
            if bigq:  # other instruments
                if lag_q:  # lags for other instruments
                    wq = {}
                    wqnames = {}
                    for r in range(self.n_eq):
                        bigwq = WS * bigq[r]
                        wqnameswork = ["W_" + i for i in self.name_bigq[r]]
                        wqnames[r] = wqnameswork
                        wq[r] = bigwq
                        for i in range(1, w_lags):
                            bigwq = WS * bigwq
                            wq[r] = np.hstack((wq[r], bigwq))
                            wqnameswork = ["W" + i for i in wqnameswork]
                            wqnames[r] = wqnames[r] + wqnameswork
                        bigq[r] = np.hstack((bigq[r], bigwx[r], wq[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r] + wqnames[r]

                else:  # no lags for other instruments
                    for r in range(self.n_eq):
                        bigq[r] = np.hstack((bigq[r], bigwx[r]))
                        self.name_bigq[r] = self.name_bigq[r] + wxnames[r]
            else:  # no other instruments only wx
                bigq = {}
                for r in range(self.n_eq):
                    bigq[r] = bigwx[r]
                    self.name_bigq[r] = wxnames[r]

        else:
            raise Exception("Lag order must be 1 or higher")

        BaseThreeSLS.__init__(
            self, bigy=self.bigy, bigX=bigX, bigyend=bigyend, bigq=bigq
        )

        # inference
        self.tsls_inf = sur_setp(self.b3SLS, self.varb)

        # test on joint significance of spatial coefficients
        if spat_diag:
            self.joinrho = sur_joinrho(self.n_eq, self.bigK, self.b3SLS, self.varb)
        else:
            self.joinrho = None

        # test on constancy of coefficients across equations
        if check_k(self.bigK):  # only for equal number of variables
            self.surchow = sur_chow(self.n_eq, self.bigK, self.b3SLS, self.varb)
        else:
            self.surchow = None

        # list results
        self.title = "SEEMINGLY UNRELATED REGRESSIONS (SUR) - SPATIAL LAG MODEL"
        if regimes is not None:
            self.title = "SUR - SPATIAL LAG MODEL - REGIMES"
            self.chow_regimes = {}
            varb_counter = 0
            fixed_lag = 1
            if regime_lag_sep == True:
                fixed_lag += -1
            for r in range(self.n_eq):
                counter_end = varb_counter + self.b3SLS[r].shape[0]
                self.chow_regimes[r] = REGI._chow_run(
                    len(cols2regi_dic[r]),
                    fixed_lag,
                    0,
                    len(self.regimes_set),
                    self.b3SLS[r],
                    self.varb[varb_counter:counter_end, varb_counter:counter_end],
                )
                varb_counter = counter_end
            regimes = True
        SUMMARY.SUR(
            reg=self,
            tsls=True,
            spat_diag=spat_diag,
            nonspat_diag=nonspat_diag,
            ml=False,
            regimes=regimes,
        )


def _get_spatial_lag(reg, bigyend, WS, name_bigyend):
    bigylag = {}
    for r in range(reg.n_eq):
        bigylag[r] = WS * reg.bigy[r]
    if bigyend is not None:
        for r in range(reg.n_eq):
            bigyend[r] = np.hstack((bigyend[r], bigylag[r]))
        # adjust variable names
        for r in range(reg.n_eq):
            wyname = "W_" + reg.name_bigy[r]
            name_bigyend[r].append(wyname)
    else:  # no other endogenous variables
        bigyend = {}
        for r in range(reg.n_eq):
            bigyend[r] = bigylag[r]
        # variable names
        for r in range(reg.n_eq):
            wyname = ["W_" + reg.name_bigy[r]]
            name_bigyend[r] = wyname
    return bigyend, name_bigyend


def _test():
    import doctest

    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    # _test()
    import numpy as np
    import libpysal
    from .sur_utils import sur_dictxy
    from libpysal.examples import load_example
    from libpysal.weights import Queen

    nat = load_example("Natregimes")
    db = libpysal.io.open(nat.get_path("natregimes.dbf"), "r")
    w = Queen.from_shapefile(nat.get_path("natregimes.shp"))
    w.transform = "r"
    y_var0 = ["HR80", "HR90"]
    x_var0 = [["PS80", "UE80"], ["PS90", "UE90"]]
    regimes = db.by_col("SOUTH")

    bigy0, bigX0, bigyvars0, bigXvars0 = sur_dictxy(db, y_var0, x_var0)

    reg = SURlagIV(
        bigy0,
        bigX0,
        w=w,
        regimes=regimes,
        name_bigy=bigyvars0,
        name_bigX=bigXvars0,
        name_ds="NAT",
        name_w="nat_queen",
        nonspat_diag=True,
        spat_diag=True,
        regime_lag_sep=True,
    )
    print(reg.summary)
