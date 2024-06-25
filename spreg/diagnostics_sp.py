"""
Spatial diagnostics module
"""
__author__ = "Luc Anselin lanselin@gmail.com, Daniel Arribas-Bel darribas@asu.edu, Pedro Amaral pedrovma@gmail.com"

from .utils import spdot

# from scipy.stats.stats import chisqprob
from scipy import stats

# stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from scipy.stats import norm
import numpy as np
import numpy.linalg as la

__all__ = ["LMtests", "MoranRes", "AKtest"]


class LMtests:
    """
    Lagrange Multiplier tests. Implemented as presented in :cite:`Anselin1996a` and :cite:`KoleyBera2024`

    Attributes
    ----------

    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
    tests       : list
                  Lists of strings with the tests desired to be performed.
                  Values may be:

                  * 'all': runs all the options (default)
                  * 'lme': LM error test
                  * 'rlme': Robust LM error test
                  * 'lml' : LM lag test
                  * 'rlml': Robust LM lag test
                  * 'sarma': LM SARMA test
                  * 'lmwx': LM test for WX
                  * 'rlmwx': Robust LM WX test
                  * 'lmspdurbin': Joint test for SDM
                  * 'rlmdurlag': Robust LM Lag - SDM

    Parameters
    ----------

    lme         : tuple
                  (Only if 'lme' or 'all' was in tests). Pair of statistic and
                  p-value for the LM error test.
    lml         : tuple
                  (Only if 'lml' or 'all' was in tests). Pair of statistic and
                  p-value for the LM lag test.
    rlme        : tuple
                  (Only if 'rlme' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM error test.
    rlml        : tuple
                  (Only if 'rlml' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM lag test.
    sarma       : tuple
                  (Only if 'sarma' or 'all' was in tests). Pair of statistic
                  and p-value for the SARMA test.
    lmwx       : tuple
                  (Only if 'lmwx' or 'all' was in tests). Pair of statistic
                  and p-value for the LM test for WX.
    rlmwx       : tuple
                  (Only if 'rlmwx' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM WX test.
    rlmdurlag   : tuple
                  (Only if 'rlmdurlag' or 'all' was in tests). Pair of statistic
                  and p-value for the Robust LM Lag - SDM test.
    lmspdurbin  : tuple
                  (Only if 'lmspdurbin' or 'all' was in tests). Pair of statistic
                  and p-value for the Joint test for SDM.
    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import OLS
    >>> import spreg

    Open the csv file to access the data for analysis

    >>> csv = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Pull out from the csv the files we need ('HOVAL' as dependent as well as
    'INC' and 'CRIME' as independent) and directly transform them into nx1 and
    nx2 arrays, respectively

    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T

    Create the weights object from existing .gal file

    >>> w = libpysal.io.open(libpysal.examples.get_path('columbus.gal'), 'r').read()

    Row-standardize the weight object (not required although desirable in some
    cases)

    >>> w.transform='r'

    Run an OLS regression

    >>> ols = OLS(y, x)

    Run all the LM tests in the residuals. These diagnostics test for the
    presence of remaining spatial autocorrelation in the residuals of an OLS
    model and give indication about the type of spatial model. There are five
    types: presence of a spatial lag model (simple and robust version),
    presence of a spatial error model (simple and robust version) and joint presence
    of both a spatial lag as well as a spatial error model.

    >>> lms = spreg.LMtests(ols, w)

    LM error test:

    >>> print(round(lms.lme[0],4), round(lms.lme[1],4))
    3.0971 0.0784

    LM lag test:

    >>> print(round(lms.lml[0],4), round(lms.lml[1],4))
    0.9816 0.3218

    Robust LM error test:

    >>> print(round(lms.rlme[0],4), round(lms.rlme[1],4))
    3.2092 0.0732

    Robust LM lag test:

    >>> print(round(lms.rlml[0],4), round(lms.rlml[1],4))
    1.0936 0.2957

    LM SARMA test:

    >>> print(round(lms.sarma[0],4), round(lms.sarma[1],4))
    4.1907 0.123

    LM test for WX:

    >>> print(round(lms.lmwx[0],4), round(lms.lmwx[1],4))
    1.3377 0.5123

    Robust LM WX test:

    >>> print(round(lms.rlmwx[0],4), round(lms.rlmwx[1],4))
    3.4532 0.1779

    Robust LM Lag - SDM:
    >>> print(round(lms.rlmdurlag[0],4), round(lms.rlmdurlag[1],4))
    3.0971 0.0784

    Joint test for SDM:

    >>> print(round(lms.lmspdurbin[0],4), round(lms.lmspdurbin[1],4))
    4.4348 0.2182
    """

    def __init__(self, ols, w, tests=["all"]):
        cache = spDcache(ols, w)
        if tests == ["all"]:
            tests = ["lme", "lml", "rlme", "rlml", "sarma", "lmwx", "lmspdurbin", "rlmwx",
                "rlmdurlag", "lmslxerr"]    # added back in for access
        if any(test in ["lme", "lmslxerr"] for test in tests):
        #if "lme" in tests:
            self.lme = lmErr(ols, w, cache)
        if any(test in ["lml", "rlmwx"] for test in tests):
            self.lml = lmLag(ols, w, cache)
        if "rlme" in tests:
            self.rlme = rlmErr(ols, w, cache)
        if "rlml" in tests:
            self.rlml = rlmLag(ols, w, cache)
        if "sarma" in tests:
            self.sarma = lmSarma(ols, w, cache)
        #if any(test in ["lmwx", "rlmdurlag", "lmslxerr"] for test in tests):
        if any(test in ["lmwx", "rlmdurlag","lmslxerr"] for test in tests):
            self.lmwx = lm_wx(ols, w)
        if any(test in ["lmspdurbin", "rlmdurlag", "rlmwx"] for test in tests):
            self.lmspdurbin = lm_spdurbin(ols, w)
        if "rlmwx" in tests:
            self.rlmwx = rlm_wx(ols, self.lmspdurbin, self.lml)
        if "rlmdurlag" in tests:
            self.rlmdurlag = rlm_durlag(self.lmspdurbin, self.lmwx)
        if "lmslxerr" in tests: #currently removed - LA added back in for access
            self.lmslxerr = lm_slxerr(ols, self.lme, self.lmwx)

class MoranRes:
    """
    Moran's I for spatial autocorrelation in residuals from OLS regression


    Parameters
    ----------
    ols         : OLS
                  OLS regression object
    w           : W
                  Spatial weights instance
    z           : boolean
                  If set to True computes attributes eI, vI and zI. Due to computational burden of vI, defaults to False.

    Attributes
    ----------
    I           : float
                  Moran's I statistic
    eI          : float
                  Moran's I expectation
    vI          : float
                  Moran's I variance
    zI          : float
                  Moran's I standardized value

    Examples
    --------

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import OLS
    >>> import spreg

    Open the csv file to access the data for analysis

    >>> csv = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'),'r')

    Pull out from the csv the files we need ('HOVAL' as dependent as well as
    'INC' and 'CRIME' as independent) and directly transform them into nx1 and
    nx2 arrays, respectively

    >>> y = np.array([csv.by_col('HOVAL')]).T
    >>> x = np.array([csv.by_col('INC'), csv.by_col('CRIME')]).T

    Create the weights object from existing .gal file

    >>> w = libpysal.io.open(libpysal.examples.get_path('columbus.gal'), 'r').read()

    Row-standardize the weight object (not required although desirable in some
    cases)

    >>> w.transform='r'

    Run an OLS regression

    >>> ols = OLS(y, x)

    Run Moran's I test for residual spatial autocorrelation in an OLS model.
    This computes the traditional statistic applying a correction in the
    expectation and variance to account for the fact it comes from residuals
    instead of an independent variable

    >>> m = spreg.MoranRes(ols, w, z=True)

    Value of the Moran's I statistic:

    >>> print(round(m.I,4))
    0.1713

    Value of the Moran's I expectation:

    >>> print(round(m.eI,4))
    -0.0345

    Value of the Moran's I variance:

    >>> print(round(m.vI,4))
    0.0081

    Value of the Moran's I standardized value. This is
    distributed as a standard Normal(0, 1)

    >>> print(round(m.zI,4))
    2.2827

    P-value of the standardized Moran's I value (z):

    >>> print(round(m.p_norm,4))
    0.0224
    """

    def __init__(self, ols, w, z=False):
        cache = spDcache(ols, w)
        self.I = get_mI(ols, w, cache)
        if z:
            self.eI = get_eI(ols, w, cache)
            self.vI = get_vI(ols, w, self.eI, cache)
            self.zI, self.p_norm = get_zI(self.I, self.eI, self.vI)


class AKtest:
    """
    Moran's I test of spatial autocorrelation for IV estimation.
    Implemented following the original reference :cite:`Anselin1997`


    Parameters
    ----------

    iv          : TSLS
                  Regression object from TSLS class
    w           : W
                  Spatial weights instance
    case        : string
                  Flag for special cases (default to 'nosp'):

                  * 'nosp': Only NO spatial end. reg.
                  * 'gen': General case (spatial lag + end. reg.)

    Attributes
    ----------

    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals
                  :math:`ak = \dfrac{N \times I^*}{\phi^2}`.
                  Note: if case='nosp' then it simplifies to the LMerror
    p           : float
                  P-value of the test

    Examples
    --------

    We first need to import the needed modules. Numpy is needed to convert the
    data we read into arrays that ``spreg`` understands and ``pysal`` to
    perform all the analysis. The TSLS is required to run the model on
    which we will perform the tests.

    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import TSLS, GM_Lag, AKtest

    Open data on Columbus neighborhood crime (49 areas) using libpysal.io.open().
    This is the DBF associated with the Columbus shapefile.  Note that
    libpysal.io.open() also reads data in CSV format; since the actual class
    requires data to be passed in as numpy arrays, the user can read their
    data in using any method.

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),'r')

    Before being able to apply the diagnostics, we have to run a model and,
    for that, we need the input variables. Extract the CRIME column (crime
    rates) from the DBF file and make it the dependent variable for the
    regression. Note that PySAL requires this to be an numpy array of shape
    (n, 1) as opposed to the also common shape of (n, ) that other packages
    accept.

    >>> y = np.array(db.by_col("CRIME"))
    >>> y = np.reshape(y, (49,1))

    Extract INC (income) vector from the DBF to be used as
    independent variables in the regression.  Note that PySAL requires this to
    be an nxj numpy array, where j is the number of independent variables (not
    including a constant). By default this model adds a vector of ones to the
    independent variables passed in, but this can be overridden by passing
    constant=False.

    >>> X = []
    >>> X.append(db.by_col("INC"))
    >>> X = np.array(X).T

    In this case, we consider HOVAL (home value) as an endogenous regressor,
    so we acknowledge that by reading it in a different category.

    >>> yd = []
    >>> yd.append(db.by_col("HOVAL"))
    >>> yd = np.array(yd).T

    In order to properly account for the endogeneity, we have to pass in the
    instruments. Let us consider DISCBD (distance to the CBD) is a good one:

    >>> q = []
    >>> q.append(db.by_col("DISCBD"))
    >>> q = np.array(q).T

    Now we are good to run the model. It is an easy one line task.

    >>> reg = TSLS(y, X, yd, q=q)

    Now we are concerned with whether our non-spatial model presents spatial
    autocorrelation in the residuals. To assess this possibility, we can run
    the Anselin-Kelejian test, which is a version of the classical LM error
    test adapted for the case of residuals from an instrumental variables (IV)
    regression. First we need an extra object, the weights matrix, which
    includes the spatial configuration of the observations
    into the error component of the model. To do that, we can open an already
    existing gal file or create a new one. In this case, we will create one
    from ``columbus.shp``.

    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))

    Unless there is a good reason not to do it, the weights have to be
    row-standardized so every row of the matrix sums to one. Among other
    things, this allows to interpret the spatial lag of a variable as the
    average value of the neighboring observations. In PySAL, this can be
    easily performed in the following way:

    >>> w.transform = 'r'

    We are good to run the test. It is a very simple task:

    >>> ak = AKtest(reg, w)

    And explore the information obtained:

    >>> print('AK test: %f\tP-value: %f'%(ak.ak, ak.p))
    AK test: 4.642895      P-value: 0.031182

    The test also accomodates the case when the residuals come from an IV
    regression that includes a spatial lag of the dependent variable. The only
    requirement needed is to modify the ``case`` parameter when we call
    ``AKtest``. First, let us run a spatial lag model:

    >>> reg_lag = GM_Lag(y, X, yd, q=q, w=w)

    And now we can run the AK test and obtain similar information as in the
    non-spatial model.

    >>> ak_sp = AKtest(reg, w, case='gen')
    >>> print('AK test: %f\tP-value: %f'%(ak_sp.ak, ak_sp.p))
    AK test: 1.157593      P-value: 0.281965

    """

    def __init__(self, iv, w, case="nosp"):
        if case == "gen":
            cache = spDcache(iv, w)
            self.mi, self.ak, self.p = akTest(iv, w, cache)
        elif case == "nosp":
            cache = spDcache(iv, w)
            self.mi = get_mI(iv, w, cache)
            self.ak, self.p = lmErr(iv, w, cache)
        else:
            print(
                """\n
            Fix the optional argument 'case' to match the requirements:
                * 'gen': General case (spatial lag + end. reg.)
                * 'nosp': No spatial end. reg.
            \n"""
            )


class spDcache:
    """
    Helper class to compute reusable pieces in the spatial diagnostics module
    ...

    Parameters
    ----------

    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance

    Attributes
    ----------

    j           : array
                  1x1 array with the result from:
                  :math:`J = \dfrac{1}{[(WX\beta)' M (WX\beta) + T \sigma^2]}`
    wu          : array
                  nx1 array with spatial lag of the residuals
    utwuDs      : array
                  1x1 array with the result from:
                  :math:`utwuDs = \dfrac{u' W u}{\tilde{\sigma^2}}`
    utwyDs      : array
                  1x1 array with the result from:
                  :math:`utwyDs = \dfrac{u' W y}{\tilde{\sigma^2}}`
    t           : array
                  1x1 array with the result from :
                  :math:` T = tr[(W' + W) W]`
    trA         : float
                  Trace of A as in Cliff & Ord (1981)

    """

    def __init__(self, reg, w):
        self.reg = reg
        self.w = w
        self._cache = {}

    @property
    def j(self):
        if "j" not in self._cache:
            wxb = self.w.sparse * self.reg.predy
            wxb2 = np.dot(wxb.T, wxb)
            xwxb = spdot(self.reg.x.T, wxb)
            num1 = wxb2 - np.dot(xwxb.T, np.dot(self.reg.xtxi, xwxb))
            num = num1 + (self.t * self.reg.sig2n)
            den = self.reg.n * self.reg.sig2n
            self._cache["j"] = num / den
        return self._cache["j"]

    @property
    def wu(self):
        if "wu" not in self._cache:
            self._cache["wu"] = self.w.sparse * self.reg.u
        return self._cache["wu"]

    @property
    def utwuDs(self):
        if "utwuDs" not in self._cache:
            res = np.dot(self.reg.u.T, self.wu) / self.reg.sig2n
            self._cache["utwuDs"] = res
        return self._cache["utwuDs"]

    @property
    def utwyDs(self):
        if "utwyDs" not in self._cache:
            res = np.dot(self.reg.u.T, self.w.sparse * self.reg.y)
            self._cache["utwyDs"] = res / self.reg.sig2n
        return self._cache["utwyDs"]

    @property
    def t(self):
        if "t" not in self._cache:
            prod = (self.w.sparse.T + self.w.sparse) * self.w.sparse
            self._cache["t"] = np.sum(prod.diagonal())
        return self._cache["t"]

    @property
    def trA(self):
        if "trA" not in self._cache:
            xtwx = spdot(self.reg.x.T, spdot(self.w.sparse, self.reg.x))
            mw = np.dot(self.reg.xtxi, xtwx)
            self._cache["trA"] = np.sum(mw.diagonal())
        return self._cache["trA"]

    @property
    def AB(self):
        """
        Computes A and B matrices as in Cliff-Ord 1981, p. 203
        """
        if "AB" not in self._cache:
            U = (self.w.sparse + self.w.sparse.T) / 2.0
            z = spdot(U, self.reg.x, array_out=False)
            c1 = spdot(self.reg.x.T, z, array_out=False)
            c2 = spdot(z.T, z, array_out=False)
            G = self.reg.xtxi
            A = spdot(G, c1)
            B = spdot(G, c2)
            self._cache["AB"] = [A, B]
        return self._cache["AB"]


def lmErr(reg, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (9) of Anselin et al.
    (1996) :cite:`Anselin1996a`.

    Attributes
    ----------
    reg         : OLS_dev, TSLS_dev, STSLS_dev
                  Instance from a regression class
    w           : W
                  Spatial weights instance
    spDcache    : spDcache
                  Instance of spDcache class

    Returns
    -------
    lme         : tuple
                  Pair of statistic and p-value for the LM error test.

    """
    lm = spDcache.utwuDs ** 2 / spDcache.t
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lmLag(ols, w, spDcache):
    """
    LM lag test. Implemented as presented in eq. (13) of Anselin et al.
    (1996) :cite:`Anselin1996a`.

    Attributes
    ----------
    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache     : spDcache
                   Instance of spDcache class

    Returns
    -------
    lml         : tuple
                  Pair of statistic and p-value for the LM lag test.

    """
    lm = spDcache.utwyDs ** 2 / (ols.n * spDcache.j)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlmErr(ols, w, spDcache):
    """
    Robust LM error test. Implemented as presented in eq. (8) of Anselin et
    al. (1996) :cite:`Anselin1996a`.

    NOTE: eq. (8) has an errata, the power -1 in the denominator
    should be inside the square bracket.

    Attributes
    ----------
    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache    : spDcache
                  Instance of spDcache class

    Returns
    -------
    rlme        : tuple
                  Pair of statistic and p-value for the Robust LM error test.

    """
    nj = ols.n * spDcache.j
    num = (spDcache.utwuDs - (spDcache.t * spDcache.utwyDs) / nj) ** 2
    den = spDcache.t * (1.0 - (spDcache.t / nj))
    lm = num / den
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def rlmLag(ols, w, spDcache):
    """
    Robust LM lag test. Implemented as presented in eq. (12) of Anselin et al.
    (1996) :cite:`Anselin1996a`.

    Attributes
    ----------
    ols             : OLS_dev
                      Instance from an OLS_dev regression
    w               : W
                      Spatial weights instance
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------
    rlml            : tuple
                      Pair of statistic and p-value for the Robust LM lag test.

    """
    lm = (spDcache.utwyDs - spDcache.utwuDs) ** 2 / ((ols.n * spDcache.j) - spDcache.t)
    pval = chisqprob(lm, 1)
    return (lm[0][0], pval[0][0])


def lmSarma(ols, w, spDcache):
    """
    LM error test. Implemented as presented in eq. (15) of Anselin et al.
    (1996) :cite:`Anselin1996a`.

    Attributes
    ----------
    ols         : OLS_dev
                  Instance from an OLS_dev regression
    w           : W
                  Spatial weights instance
    spDcache     : spDcache
                  Instance of spDcache class

    Returns
    -------
    sarma       : tuple
                  Pair of statistic and p-value for the LM sarma test.

    """

    first = (spDcache.utwyDs - spDcache.utwuDs) ** 2 / (w.n * spDcache.j - spDcache.t)
    secnd = spDcache.utwuDs ** 2 / spDcache.t
    lm = first + secnd
    pval = chisqprob(lm, 2)
    return (lm[0][0], pval[0][0])

def lm_wx(reg, w):
    """
    LM test for WX. Implemented as presented in Koley & Bera (2024) :cite:`KoleyBera2024`.

    Attributes
    ----------
    reg         : OLS
                  Instance from an OLS regression
    w           : W
                  Spatial weights instance

    Returns
    -------
    lmwx        : tuple
                  Pair of statistic and p-value for the LM test for WX.

    """

    # preliminaries
    # set up X1 (constant) and X (no constant) as x1 and xx
    x1 = reg.x
    xx = x1[:,1:]
    # WX
    wx = w.sparse * xx
    # end of preliminaries
    # X'W'u
    xtwtu = wx.T @ reg.u
    # X'W'X1(X1'X1)-1X1WX
    mx1 = wx.T @ x1
    mx = (mx1 @ reg.xtxi) @ mx1.T
    xwwx = wx.T @ wx
    xqx = xwwx - mx
    xqxi = la.inv(xqx)
    # RSgamma: (X'W'u)'(X'Q1X)-1(X'W'u) / sig2n
    xpwpu = wx.T @ reg.u
    rsg1 = (xpwpu.T @ xqxi) @ xpwpu
    rsgam = rsg1[0][0] / reg.sig2n
    pval = chisqprob(rsgam, (reg.k - 1))
    rsgamma = (rsgam,pval)
    return(rsgamma)

def lm_spdurbin(reg,w):
    """
    Joint test for SDM. Implemented as presented in Koley & Bera (2024) :cite:`KoleyBera2024`.

    Attributes
    ----------
    reg         : OLS
                  Instance from an OLS regression
    w           : W
                  Spatial weights instance

    Returns
    -------
    lmspdurbin  : tuple
                  Pair of statistic and p-value for the Joint test for SDM.

    """

    # preliminaries
    # set up X1 (constant) and X (no constant) as x1 and xx
    x1 = reg.x
    xx = x1[:,1:]
    k = x1.shape[1]
    # WX
    wx = w.sparse * xx
    # X1b
    xb = reg.predy
    # WX1b
    wxb = w.sparse * xb
    # Wy
    wy = w.sparse * reg.y
    # y'W'e / sig2n
    drho = (wy.T @ reg.u) / reg.sig2n
    # X'W'e / sign2n
    dgam = (wx.T @ reg.u) / reg.sig2n
    # P = T = tr(W2 + W'W)
    pp = w.trcWtW_WW
    # end of preliminaries
    # J_11: block matrix with X1'X1 and n/2sig2n
    jj1a = np.hstack((reg.xtx,np.zeros((k,1))))
    jj1b = np.hstack((np.zeros((1,k)),np.array([reg.n/(2.0*reg.sig2n)]).reshape(1,1)))
    jj11 = np.vstack((jj1a,jj1b))
    # J_12: matrix with k-1 rows X1'WX1b and X1'WX, and 1 row of zeros
    jj12a = np.hstack((x1.T @ wxb, x1.T @ wx))
    jj12 = np.vstack((jj12a,np.zeros((1,k))))
    # J_22 matrix with diagonal elements b'X1'W'WX1b + T.sig2n and X'W'WX
    # and off-diagonal element b'X1'W'WX
    jj22a = wxb.T @ wxb + pp * reg.sig2n
    jj22a = jj22a.reshape(1,1)
    wxbtwx = (wxb.T @ wx).reshape(1,k-1)
    jj22b = np.hstack((jj22a,wxbtwx))
    wxtwx = wx.T @ wx
    jj22c = np.hstack((wxbtwx.T,wxtwx))
    jj22 = np.vstack((jj22b,jj22c))
    # J^22 (the inverse) from J^22 = (J_22 - J_21.J_11^-1.J_12)^-1
    jj11i = la.inv(jj11)
    j121121 = (jj12.T @ jj11i) @ jj12
    jj22i1 = jj22 - j121121
    jj22i = la.inv(jj22i1)
    # rescale by sig2n
    jj22i = jj22i * reg.sig2n
    # statistic
    dd = np.vstack((drho,dgam))
    rsjoint = (dd.T @ jj22i) @ dd
    rsjoint = rsjoint[0][0]
    pval = chisqprob(rsjoint, k)
    rsrhogam = (rsjoint, pval)
    return(rsrhogam)

def rlm_wx(reg,lmspdurbin,lmlag):
    """
    Robust LM WX test. Implemented as presented in Koley & Bera (2024) :cite:`KoleyBera2024`.

    Attributes
    ----------
    reg         : OLS
                  Instance from an OLS regression
    lmspdurbin  : tuple
                  Joint test for SDM as in lm_spdurbin function
    lmlag       : tuple
                  LM Lag test as in lmLag function

    Returns
    -------
    rlmwx       : tuple
                  Pair of statistic and p-value for the Robust LM WX test.

    """
    # robust gamma = rsjoint - rsrho
    rsgams = lmspdurbin[0] - lmlag[0]
    pval = chisqprob(rsgams,(reg.k - 1))
    rsgamstar = (rsgams, pval)
    return(rsgamstar)

def rlm_durlag(lmspdurbin,lmwx):
    """
    Robust LM Lag - SDM. Implemented as presented in Koley & Bera (2024) :cite:`KoleyBera2024`.

    Attributes
    ----------
    lmspdurbin  : tuple
                  Joint test for SDM as in lm_spdurbin function
    lmwx        : tuple
                  LM test for WX as in lm_wx function

    Returns
    -------
    rlmwx       : tuple
                  Pair of statistic and p-value for the Robust LM Lag - SDM test.
    """

    # robust rho = rsjoint - rsgam
    rsrhos = lmspdurbin[0] - lmwx[0]
    pval = chisqprob(rsrhos,1)
    rsrhostar = (rsrhos, pval)
    return(rsrhostar)

def lm_slxerr(reg,lmerr,lmwx):
    """
    Joint test for Error and WX. Implemented as presented in Koley & Bera (2024) :cite:`KoleyBera2024`.

    Attributes
    ----------
    reg         : OLS
                  Instance from an OLS regression
    lmerr         : tuple
                  LM Error test as in lmErr function
    lmwx        : tuple
                  LM test for WX as in lm_wx function

    Returns
    -------
    rlmwx       : tuple
                  Pair of statistic and p-value for the Joint test for Error and WX.
    """
    rslamgam = lmerr[0] + lmwx[0]
    pval = chisqprob(rslamgam,reg.k)
    rslamgamma = (rslamgam,pval)
    return(rslamgamma)

def get_mI(reg, w, spDcache):
    """
    Moran's I statistic of spatial autocorrelation as showed in Cliff & Ord
    (1981) :cite:`clifford1981`, p. 201-203

    Attributes
    ----------
    reg             : OLS_dev, TSLS_dev, STSLS_dev
                      Instance from a regression class
    w               : W
                      Spatial weights instance
    spDcache        : spDcache
                      Instance of spDcache class

    Returns
    -------
    moran           : float
                      Statistic Moran's I test.

    """
    mi = (w.n * np.dot(reg.u.T, spDcache.wu)) / (w.s0 * reg.utu)
    return mi[0][0]


def get_vI(ols, w, ei, spDcache):
    """
    Moran's I variance coded as in :cite:`clifford1981` (p. 201-203) and R's spdep
    """
    A = spDcache.AB[0]
    trA2 = np.dot(A, A)
    trA2 = np.sum(trA2.diagonal())

    B = spDcache.AB[1]
    trB = np.sum(B.diagonal()) * 4.0
    vi = (w.n ** 2 / (w.s0 ** 2 * (w.n - ols.k) * (w.n - ols.k + 2.0))) * (
            w.s1 + 2.0 * trA2 - trB - ((2.0 * (spDcache.trA ** 2)) / (w.n - ols.k))
    )
    return vi


def get_eI(ols, w, spDcache):
    """
    Moran's I expectation using matrix M
    """
    return -(w.n * spDcache.trA) / (w.s0 * (w.n - ols.k))


def get_zI(I, ei, vi):
    """
    Standardized I

    Returns two-sided p-values as provided in the GeoDa family
    """
    z = abs((I - ei) / np.sqrt(vi))
    pval = norm.sf(z) * 2.0
    return (z, pval)


def akTest(iv, w, spDcache):
    """
    Computes AK-test for the general case (end. reg. + sp. lag)

    Parameters
    ----------

    iv          : STSLS_dev
                  Instance from spatial 2SLS regression
    w           : W
                  Spatial weights instance
    spDcache    : spDcache
                  Instance of spDcache class

    Attributes
    ----------
    mi          : float
                  Moran's I statistic for IV residuals
    ak          : float
                  Square of corrected Moran's I for residuals:
                  :math:`ak = \dfrac{N \times I^*}{\phi^2}`
    p           : float
                  P-value of the test

    """
    mi = get_mI(iv, w, spDcache)
    # Phi2
    etwz = spdot(iv.u.T, spdot(w.sparse, iv.z))
    a = np.dot(etwz, np.dot(iv.varb, etwz.T))
    s12 = (w.s0 / w.n) ** 2
    phi2 = (spDcache.t + (4.0 / iv.sig2n) * a) / (s12 * w.n)
    ak = w.n * mi ** 2 / phi2
    pval = chisqprob(ak, 1)
    return (mi, ak[0][0], pval[0][0])


def comfac_test(rho, beta, gamma, vm):
    """
    Computes the Spatial Common Factor Hypothesis test as shown in Anselin (1988, p. 226-229).
    Note that for the Common Factor Hypothesis test to be valid, gamma has to equal
    *negative* rho times beta for all beta parameters.
    That is, when rho is positive, a positive beta means gamma must be negative and vice versa.
    For a negative rho, beta, and gamma must have the same sign.
    If those signs are not compatible, the test will not be meaningful.

    Parameters
    ----------

    rho         : float
                  Spatial autoregressive coefficient (as in rho*Wy)
    beta        : array
                  Coefficients of the exogenous (not spatially lagged) variables, without the constant (as in X*beta)
    gamma       : array
                  coefficients of the spatially lagged exogenous variables (as in WX*gamma)
    vm          : array
                  Variance-covariance matrix of the coefficients
                  Obs. Needs to match the order of theta' = [beta', gamma', lambda]

    Returns
    -------
    W       : float
              Wald statistic
    pvalue  : float
              P value for Wald statistic calculated as a Chi sq. distribution
              with k-1 degrees of freedom

    """
    g = rho * beta + gamma
    G = np.vstack((rho * np.eye(beta.shape[0]), np.eye(beta.shape[0]), beta.T))

    GVGi = la.inv(np.dot(G.T, np.dot(vm, G)))
    W = np.dot(g.T, np.dot(GVGi, g))[0][0]
    df = G.shape[1]
    pvalue = chisqprob(W, df)
    return W, pvalue


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
