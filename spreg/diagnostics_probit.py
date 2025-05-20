"""
Diagnostics in probit regression. 
        
"""
__author__ = (
    "Luc Anselin lanselin@gmail.com, Pedro Amaral pedrovma@gmail.com "
)

from math import sqrt, pi

from libpysal.common import MISSINGVALUE
import numpy as np
import numpy.linalg as la
import scipy.sparse as SP
from scipy import stats
from scipy.stats import norm

__all__ = [
    "pred_table",
    "probit_fit",
    "probit_lrtest",
    "mcfad_rho",
    "probit_ape",
    "sp_tests",
    "moran_KP",
]


def pred_table(reg):
    """
    Calculates a table comparing predicted to actual outcomes for a 
    discrete choice model

    Parameters
    ----------
    reg             : regression object
                      output instance from a probit regression model

    Returns
    ----------
    predtab_vals    : dictionary
                      includes margins and cells of actual and predicted
                      values for discrete choice model
                      actpos   : observed positives (=1)
                      actneg   : observed negatives (=0)
                      predpos  : predicted positives 
                      predneg  : predicted negatives
                      truepos  : predicted 1 when actual = 1
                      falsepos : predicted 1 when actual = 0
                      trueneg  : predicted 0 when actual = 0
                      falseneg : predicted 0 when actual = 1

    """
    predtab_vals = {}
    pos = reg.y.sum()
    predtab_vals["actpos"] = int(pos)
    neg = reg.n - pos
    predtab_vals["actneg"] = int(neg)
    act1 = (reg.y == 1) * 1
    act0 = (reg.y == 0) * 1
    ppos = reg.predybin.sum()
    predtab_vals["predpos"] = ppos
    pneg = reg.n - ppos
    predtab_vals["predneg"] = pneg
    pred1 = (reg.predybin == 1) * 1
    pred0 = (reg.predybin == 0) * 1
    truep = (pred1 * act1) * 1
    predtab_vals["truepos"] = truep.sum()
    truen = (pred0 * act0) * 1
    predtab_vals["trueneg"] = truen.sum()
    fpos = (pred1 * act0) * 1
    predtab_vals["falsepos"] = fpos.sum()
    fneg = (pred0 * act1) * 1
    predtab_vals["falseneg"] = fneg.sum()

    return predtab_vals


def probit_fit(reg):
    """
    Various measures of fit for discrete choice models, derived from the
    prediction table (pred_table)
    
    Parameters
    ----------
    reg             : regression object
                      output instance from a probit regression model
                      must contain predtable attribute

    Returns
    ----------
    prob_fit    : a dictionary containing various measures of fit
                  TPR    : true positive rate (sensitivity, recall, hit rate)
                  TNR    : true negative rate (specificity, selectivity)
                  PREDPC : accuracy, percent correctly predicted
                  BA     : balanced accuracy
    
    """

    prob_fit = {}
    prob_fit["TPR"] = 100.0 * reg.predtable["truepos"] / reg.predtable["actpos"]
    prob_fit["TNR"] = 100.0 * reg.predtable["trueneg"] / reg.predtable["actneg"]
    prob_fit["BA"] = (prob_fit["TPR"] + prob_fit["TNR"])/2.0
    prob_fit["PREDPC"] = 100.0 * (reg.predtable["truepos"] + reg.predtable["trueneg"]) / reg.n

    return prob_fit

def probit_lrtest(regprob):
    """
    Likelihood ratio test statistic for probit model

    Parameters
    ----------
    regprob      : probit regression object

    Returns
    -------

    likratio     : dictionary
                   contains the statistic for the null model (L0), the LR test(likr), 
                   the degrees of freedom (df) and the p-value (pvalue)
    L0           : float
                   log likelihood of null model
    likr         : float
                   likelihood ratio statistic
    df           : integer
                   degrees of freedom
    p-value      : float
                   p-value
    """

    likratio = {}
    P = np.mean(regprob.y)
    L0 = regprob.n * (P * np.log(P) + (1 - P) * np.log(1 - P))
    likratio["L0"] = L0
    LR = -2.0 * (L0 - regprob.logl)
    likratio["likr"] = LR
    likratio["df"] = regprob.k
    pval = stats.chisqprob(LR, regprob.k)
    likratio["p-value"] = pval

    return likratio

def mcfad_rho(regprob):
    """
    McFadden's rho measure of fit

    Parameters
    ---------
    regprob    : probit regression object

    Returns
    -------
    rho        : McFadden's rho (1 - L/L0)
    
    """

    rho = 1.0 - (regprob.logl / regprob.L0)
    return rho

def probit_ape(regprob):
    """
    Average partial effects

    Parameters
    ----------
    regprob   : probit regression object

    Returns
    -------
    tuple with:
        scale          : the scale of the marginal effects, determined by regprob.scalem
                         Default: 'phimean' (Mean of individual marginal effects)
                         Alternative: 'xmean' (Marginal effects at variables mean)
        slopes         : marginal effects or average partial effects (not for constant)
        slopes_vm      : estimates of variance of marginal effects (not for constant)
        slopes_std_err : estimates of standard errors of marginal effects
        slopes_z_stat  : tuple with z-statistics and p-values for marginal effects
    
    """
        

    if regprob.scalem == "xmean":
        xmb = regprob.xmean.T @ regprob.betas
        scale = stats.norm.pdf(xmb)
            
    elif regprob.scalem == "phimean":
        scale = np.mean(regprob.phiy,axis=0)

    # average partial effects (no constant)
    slopes = (regprob.betas[1:,0] * scale).reshape(-1,1)

    # variance of partial effects
    xmb = regprob.xmean.T @ regprob.betas
    bxt = regprob.betas @ regprob.xmean.T
    dfdb = np.eye(regprob.k) - xmb * bxt
    slopes_vm = (scale ** 2) * ((dfdb @ regprob.vm) @ dfdb.T)

    # standard errors
    slopes_std_err = np.sqrt(slopes_vm[1:,1:].diagonal()).reshape(-1,1)

    # z-stats and p-values
    sl_zStat = slopes / slopes_std_err
    slopes_z_stat = [(sl_zStat[i,0],stats.norm.sf(abs(sl_zStat[i,0])) * 2) for i in range(len(slopes))]


    return (scale, slopes,slopes_vm[1:,1:],slopes_std_err,slopes_z_stat)


def sp_tests(regprob=None, obj_list=None):
    """
    Calculates tests for spatial dependence in Probit models

    Parameters
    ----------
    regprob     : regression object from spreg
                  output instance from a probit model
    obj_list    : list
                  list of regression elements from both libpysal and statsmodels' ProbitResults
                  The list should be such as:
                  [libpysal.weights, ProbitResults.fittedvalues, ProbitResults.resid_response, ProbitResults.resid_generalized]               
    
    Returns
    -------
    tuple with LM_Err, moran, ps as 2x1 arrays with statistic and p-value
               LM_Err: Pinkse
               moran : Kelejian-Prucha generalized Moran
               ps    : Pinkse-Slade

    Examples
    --------
    The results of this function will be automatically added to the output of the probit model if using spreg.
    If using the Probit estimator from statsmodels, the user can call the function with the obj_list argument.
    The argument obj_list should be a list with the following elements, in this order:
    [libpysal.weights, ProbitResults.fittedvalues, ProbitResults.resid_response, ProbitResults.resid_generalized]
    The function will then return and print the results of the spatial diagnostics.

    >>> import libpysal
    >>> import statsmodels.api as sm
    >>> import geopandas as gpd
    >>> from spreg.diagnostics_probit import sp_tests

    >>> columb = libpysal.examples.load_example('Columbus')
    >>> dfs = gpd.read_file(columb.get_path("columbus.shp"))
    >>> w = libpysal.weights.Queen.from_dataframe(dfs)
    >>> w.transform='r'

    >>> y = (dfs["CRIME"] > 40).astype(float)
    >>> X = dfs[["INC","HOVAL"]]
    >>> X = sm.add_constant(X)

    >>> probit_mod = sm.Probit(y, X)
    >>> probit_res = probit_mod.fit(disp=False)
    >>> LM_err, moran, ps = sp_tests(obj_list=[w, probit_res.fittedvalues, probit_res.resid_response, probit_res.resid_generalized])
    PROBIT MODEL DIAGNOSTICS FOR SPATIAL DEPENDENCE
    TEST                              DF         VALUE           PROB
    Kelejian-Prucha (error)           1          1.721           0.0852
    Pinkse (error)                    1          3.132           0.0768
    Pinkse-Slade (error)              1          2.558           0.1097

    """
    if regprob:
        w, Phi, phi, u_naive, u_gen, n = regprob.w, regprob.predy, regprob.phiy, regprob.u_naive, regprob.u_gen, regprob.n
    elif obj_list:
        w, fittedvalues, u_naive, u_gen = obj_list
        Phi = norm.cdf(fittedvalues)
        phi = norm.pdf(fittedvalues)
        n = w.n

    try:
        w = w.sparse
    except:
        w = w
        
    # Pinkse_error:
    Phi_prod = Phi * (1 - Phi)
    sig2 = np.sum((phi * phi) / Phi_prod) / n
    LM_err_num = np.dot(u_gen.T, (w @ u_gen)) ** 2
    trWW = np.sum((w @ w).diagonal())
    trWWWWp = trWW + np.sum((w @ w.T).diagonal())
    LM_err = float(1.0 * LM_err_num / (sig2 ** 2 * trWWWWp))
    LM_err = np.array([LM_err, stats.chisqprob(LM_err, 1)])
    # KP_error:
    moran = moran_KP(w, u_naive, Phi_prod)
    # Pinkse-Slade_error:
    u_std = u_naive / np.sqrt(Phi_prod)
    ps_num = np.dot(u_std.T, (w @ u_std)) ** 2
    trWpW = np.sum((w.T @ w).diagonal())
    ps = float(ps_num / (trWW + trWpW))
    # chi-square instead of bootstrap.
    ps = np.array([ps, stats.chisqprob(ps, 1)])

    if obj_list:
        from .output import _probit_out
        reg_simile = type('reg_simile', (object,), {})()
        reg_simile.Pinkse_error = LM_err
        reg_simile.KP_error = moran
        reg_simile.PS_error = ps
        print("PROBIT MODEL "+_probit_out(reg_simile, spat_diag=True, sptests_only=True)[1:])

    return LM_err, moran, ps

def moran_KP(w, u, sig2i):
    """
    Calculates Kelejian-Prucha Moran-flavoured tests

    Parameters
    ----------

    w           : W
                  PySAL weights instance aligned with y
    u           : array
                  nx1 array of naive residuals
    sig2i       : array
                  nx1 array of individual variance

    Returns
    -------
    moran       : array, Kelejian-Prucha Moran's I with p-value   
    """
    try:
        w = w.sparse
    except:
        pass
    moran_num = np.dot(u.T, (w @ u))
    E = SP.lil_matrix(w.shape)
    E.setdiag(sig2i.flat)
    E = E.asformat("csr")
    WE = w @ E
    moran_den = np.sqrt(np.sum((WE @ WE + (w.T @ E) @ WE).diagonal()))
    moran = float(1.0 * moran_num / moran_den)
    moran = np.array([moran, stats.norm.sf(abs(moran)) * 2.0])
    return moran


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
