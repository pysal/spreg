"""Internal helper files for user output."""

__author__ = "Luc Anselin, Pedro V. Amaral"

import textwrap as TW
import numpy as np
import pandas as pd
from . import diagnostics as diagnostics
from . import diagnostics_tsls as diagnostics_tsls
from . import diagnostics_sp as diagnostics_sp
from .sputils import _sp_effects, spmultiplier

__all__ = []

###############################################################################
############### Primary functions for running summary diagnostics #############
###############################################################################

def output(reg, vm, other_end=False, robust=False, latex=False):
    strSummary = output_start(reg)
    for eq in reg.output['equation'].unique():
        try:
            reg.multi[eq].__summary = {}
            strSummary, reg.multi[eq] = out_part_top(strSummary, reg, eq)
        except:
            eq = None
            strSummary, reg = out_part_top(strSummary, reg, eq)
        strSummary, reg = out_part_middle(strSummary, reg, robust, m=eq, latex=latex)
    strSummary, reg = out_part_end(strSummary, reg, vm, other_end, m=eq)
    reg.summary = strSummary
    reg._var_type = reg.output['var_type']
    reg.output.sort_values(by=['equation', 'regime'], inplace=True)
    reg.output.drop(['var_type', 'regime', 'equation'], axis=1, inplace=True)

def output_start(reg):
    reg.__summary = {}
    strSummary = "REGRESSION RESULTS\n"
    strSummary += "------------------\n"
    reg.output = reg.output.assign(coefficients=[None] * len(reg.output), std_err=[None] * len(reg.output),
                                   zt_stat=[None] * len(reg.output), prob=[None] * len(reg.output))
    return strSummary

def out_part_top(strSummary, reg, m):
    # Top part of summary output.
    # m = None for single models, m = 1,2,3... for multiple equation models
    if m == None:
        _reg = reg  # _reg = local object with regression results
    else:
        _reg = reg.multi[m]  # _reg = local object with equation specific regression results
    title = "\nSUMMARY OF OUTPUT: " + _reg.title + "\n"
    strSummary += title
    strSummary += "-" * (len(title) - 2) + "\n"
    strSummary += "%-20s:%12s\n" % ("Data set", _reg.name_ds)
    try:
        strSummary += "%-20s:%12s\n" % ("Weights matrix", _reg.name_w)
    except:
        pass

    strSummary += "%-20s:%12s                %-22s:%12d\n" % (
        "Dependent Variable",
        _reg.name_y,
        "Number of Observations",
        _reg.n,
    )

    if not 'Probit' in _reg.__class__.__name__:
        strSummary += "%-20s:%12.4f                %-22s:%12d\n" % (
            "Mean dependent var",
            _reg.mean_y,
            "Number of Variables",
            _reg.k,
        )
        strSummary += "%-20s:%12.4f                %-22s:%12d\n" % (
            "S.D. dependent var",
            _reg.std_y,
            "Degrees of Freedom",
            _reg.n - _reg.k,
        )

    _reg.std_err = diagnostics.se_betas(_reg)
    if 'OLS' in _reg.__class__.__name__:
        _reg.t_stat = diagnostics.t_stat(_reg)
        _reg.r2 = diagnostics.r2(_reg)
        _reg.ar2 = diagnostics.ar2(_reg)
        strSummary += "%-20s:%12.4f\n%-20s:%12.4f\n" % (
            "R-squared",
            _reg.r2,
            "Adjusted R-squared",
            _reg.ar2,
        )
        _reg.__summary["summary_zt"] = "t"
    else:
        _reg.z_stat = diagnostics.t_stat(_reg, z_stat=True)
        if 'NSLX' not in _reg.__class__.__name__ and 'Probit' not in _reg.__class__.__name__:
            _reg.pr2 = diagnostics_tsls.pr2_aspatial(_reg)
            strSummary += "%-20s:%12.4f\n" % ("Pseudo R-squared", _reg.pr2)
        _reg.__summary["summary_zt"] = "z"

    try:  # Adding additional top part if there is one
        strSummary += _reg.other_top
    except:
        pass

    return (strSummary, _reg)

def out_part_middle(strSummary, reg, robust, m=None, latex=False):
    # Middle part of summary output.
    # m = None for single models, m = 1,2,3... for multiple equation models
    if m==None:
        _reg = reg #_reg = local object with regression results
        m = reg.output['equation'].unique()[0]
    else:
        _reg = reg.multi[m] #_reg = local object with equation specific regression results
    coefs = pd.DataFrame(_reg.betas, columns=['coefficients'])
    coefs['std_err'] = pd.DataFrame(_reg.std_err)
    try:
        coefs = pd.concat([coefs, pd.DataFrame(_reg.z_stat, columns=['zt_stat', 'prob'])], axis=1)
    except AttributeError:
        coefs = pd.concat([coefs, pd.DataFrame(_reg.t_stat, columns=['zt_stat', 'prob'])], axis=1)
    coefs.index = reg.output[reg.output['equation'] == m].index
    reg.output.update(coefs)
    strSummary += "\n"
    if robust:
        if robust == "white":
            strSummary += "White Standard Errors\n"
        elif robust == "hac":
            strSummary += "HAC Standard Errors; Kernel Weights: " + _reg.name_gwk + "\n"
        elif robust == "ogmm":
            strSummary += "Optimal GMM used to estimate the coefficients and the variance-covariance matrix\n"            
    strSummary += "------------------------------------------------------------------------------------\n"
    
    m_output = reg.output[reg.output['equation'] == m]
    if latex:
        df_1 = m_output.iloc[np.lexsort((m_output.index, m_output['regime']))]
        df_2 = df_1.loc[:, ['var_names', 'coefficients', 'std_err', 'zt_stat', 'prob']]
        df_2 = df_2.set_axis(['Variable', 'Coefficient', 'Std.Error', _reg.__summary['summary_zt']+'-Statistic', 'Prob.'], axis='columns', copy=False)
        cols = df_2.columns.difference(['Variable'])
        df_2[cols] = df_2[cols].astype(float).map(lambda x: "%12.5f" % x)
        df_2['Variable'] = df_2['Variable'].str.replace("_", r"\_").str.replace("%", r"\%")
        df_inlatex = df_2.style.hide(axis='index').to_latex(hrules=True)
        strSummary += df_inlatex
        strSummary += "------------------------------------------------------------------------------------\n"
    else: 
        strSummary += (
                "            Variable     Coefficient       Std.Error     %1s-Statistic     Probability\n"
                % (_reg.__summary["summary_zt"])
        )
        strSummary += "------------------------------------------------------------------------------------\n"

        for row in m_output.iloc[np.lexsort((m_output.index, m_output['regime']))].itertuples():
            try:
                strSummary += "%20s    %12.5f    %12.5f    %12.5f    %12.5f\n" % (
                    row.var_names,
                    row.coefficients,
                    row.std_err,
                    row.zt_stat,
                    row.prob
                )
            except TypeError:  # special case for models that do not have inference on the lambda term
                strSummary += "%20s    %12.5f    \n" % (
                    row.var_names,
                    row.coefficients
                )
        strSummary += "------------------------------------------------------------------------------------\n"

    try:  # Adding info on instruments if they are present
        name_q = _reg.name_q
        name_yend = _reg.name_yend
        insts = "Instruments: "
        for name in sorted(name_q):
            insts += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        insts = text_wrapper.fill(insts[:-2])
        insts += "\n"
        inst2 = "Instrumented: "
        for name in sorted(name_yend):
            inst2 += name + ", "
        text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
        inst2 = text_wrapper.fill(inst2[:-2])
        inst2 += "\n"
        inst2 += insts
        strSummary += inst2
    except:
        pass

    try:  # Adding info on regimes if they are present
        strSummary += ("Regimes variable: %s\n" % _reg.name_regimes)
        strSummary += _summary_chow(_reg)  # If local regimes present, compute Chow test.
    except:
        pass

    try:  # Adding local warning if there is one
        if _reg.warning:
            strSummary += _reg.warning
    except:
        pass

    try:  # Adding other middle part if there are any
        strSummary += _reg.other_mid
    except:
        pass

    return (strSummary, reg)

def out_part_end(strSummary, reg, vm, other_end, m=None):
    if m is not None:
        strSummary += "------------------------------------------------------------------------------------\n"
        try:  # Adding global warning if there is one
            strSummary += reg.warning
        except:
            pass
        strSummary += "\nGLOBAL DIAGNOSTICS\n"
        try:  # Adding global Chow test if there is one
            strSummary += _summary_chow(reg)
        except:
            pass
    if other_end:
        strSummary += other_end
    if vm:
        strSummary += _summary_vm(reg)
    strSummary += "================================ END OF REPORT ====================================="
    return (strSummary, reg)

def _summary_chow(reg):
    sum_text = "\nREGIMES DIAGNOSTICS\n"
    try:
        tot_SSR = reg.score
    except:
        tot_SSR = np.dot(reg.u.T, reg.u)
    try:
        n_clust = reg.nr
    except:
        n_clust = len(set(reg.clusters))
    sum_text += "%-20s:%14.2f\n%-20s:%14d\n" % (
            "Overall SSR", tot_SSR, "Number of clusters", n_clust)
    sum_text += "\n- CHOW TEST -"
    name_x_r = reg.name_x_r
    joint, regi = reg.chow.joint, reg.chow.regi
    sum_text += "\n                 VARIABLE        DF        VALUE           PROB\n"
    if reg.cols2regi == "all" or set(reg.cols2regi) == {True}:
        names_chow = name_x_r[1:]
    else:
        names_chow = [name_x_r[1:][i] for i in np.where(reg.cols2regi)[0]]
    if reg.constant_regi == "many":
        names_chow = ["CONSTANT"] + names_chow

    if 'lambda' in reg.output.var_type.values:
        if reg.output.var_type.value_counts()['lambda'] > 1:
            names_chow += ["lambda"]

    reg.output_chow = pd.DataFrame()
    reg.output_chow['var_names'] = names_chow
    reg.output_chow['df'] = reg.nr - 1
    reg.output_chow = pd.concat([reg.output_chow, pd.DataFrame(regi, columns=['value', 'prob'])], axis=1)
    reg.output_chow = pd.concat([reg.output_chow, pd.DataFrame([{'var_names': 'Global test',
                                              'df':  reg.kr * (reg.nr - 1),
                                              'value': joint[0], 'prob': joint[1]}])], ignore_index=True)
    for row in reg.output_chow.itertuples():
        sum_text += "%20s             %2d   %12.3f        %9.4f\n" % (
            row.var_names,
            row.df,
            row.value,
            row.prob
        )

    return sum_text


def _spat_diag_out(reg, w, type, moran=False, ml=False):
    strSummary = "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
    cache = diagnostics_sp.spDcache(reg, w)
    if type == "yend":
        strSummary += (
            "TEST                              DF         VALUE           PROB\n")
        if not ml:
            mi, ak, ak_p = diagnostics_sp.akTest(reg, w, cache)
            reg.ak_test = ak, ak_p
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Anselin-Kelejian Test",
                1,
                reg.ak_test[0],
                reg.ak_test[1],
            )
        if any(reg.output['var_type'] == 'rho'):
            # no common factor test if slx_vars is not "All"
            if reg.slx_lags == 1 and not any(reg.output['var_type'] == 'yend'):
                if not hasattr(reg, 'slx_vars') or not isinstance(reg.slx_vars, list):
                    wx_indices = reg.output[(reg.output['var_type'] == 'wx') & (reg.output['regime'] != '_Global')].index
                    x_indices = []
                    for m in reg.output['regime'].unique():
                        x_indices.extend(reg.output[(reg.output['regime'] == m) & (reg.output['var_type'] == 'x')].index)
                    vm_indices = x_indices + wx_indices.tolist() + reg.output[reg.output['var_type'] == 'rho'].index.tolist()
                    cft, cft_p = diagnostics_sp.comfac_test(reg.rho,
                                                        reg.betas[x_indices],
                                                        reg.betas[wx_indices],
                                                        reg.vm[vm_indices, :][:, vm_indices])
                    reg.cfh_test = cft, cft_p
                    strSummary += "%-27s    %2d   %12.3f        %9.4f\n" % (
                        "Common Factor Hypothesis Test",
                        len(wx_indices),
                        reg.cfh_test[0],
                        reg.cfh_test[1],
                    )

    elif type == "ols":
        strSummary += "- SARERR -\n"
        if not moran:
            strSummary += (
                "TEST                              DF       VALUE           PROB\n"
            )
        else:
            strSummary += (
                "TEST                           MI/DF       VALUE           PROB\n"
            )
        lm_tests = diagnostics_sp.LMtests(reg, w, tests=["lme", "lml", "rlme", "rlml", "sarma"])
        if reg.slx_lags == 0:
            try:
                lm_tests2 = diagnostics_sp.LMtests(reg, w, tests=["lmwx", "lmspdurbin", "rlmdurlag", "rlmwx","lmslxerr"])
                reg.lm_wx = lm_tests2.lmwx
                reg.lm_spdurbin = lm_tests2.lmspdurbin
                reg.rlm_wx = lm_tests2.rlmwx
                reg.rlm_durlag = lm_tests2.rlmdurlag
                reg.lm_slxerr = lm_tests2.lmslxerr #currently removed. - LA reinstated
                koley_bera = True
            except:
                koley_bera = False
        reg.lm_error = lm_tests.lme
        reg.lm_lag = lm_tests.lml
        reg.rlm_error = lm_tests.rlme
        reg.rlm_lag = lm_tests.rlml
        reg.lm_sarma = lm_tests.sarma
    

        if moran:
            moran_res = diagnostics_sp.MoranRes(reg, w, z=True)
            reg.moran_res = moran_res.I, moran_res.zI, moran_res.p_norm
            strSummary += "%-27s  %8.4f    %9.3f        %9.4f\n" % (
                "Moran's I (error)",
                reg.moran_res[0],
                reg.moran_res[1],
                reg.moran_res[2],
            )
        strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
            "Lagrange Multiplier (lag)",
            1,
            reg.lm_lag[0],
            reg.lm_lag[1],
        )
        strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
            "Robust LM (lag)",
            1,
            reg.rlm_lag[0],
            reg.rlm_lag[1],
        )
        strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
            "Lagrange Multiplier (error)",
            1,
            reg.lm_error[0],
            reg.lm_error[1],
        )
        strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
            "Robust LM (error)",
            1,
            reg.rlm_error[0],
            reg.rlm_error[1],
        )
        strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
            "Lagrange Multiplier (SARMA)",
            2,
            reg.lm_sarma[0],
            reg.lm_sarma[1],
        )
        if reg.slx_lags == 0 and koley_bera:
            strSummary += (
                "\n- Spatial Durbin -\nTEST                              DF       VALUE           PROB\n"
            )
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "LM test for WX",
                reg.k-1,
                reg.lm_wx[0],
                reg.lm_wx[1],
            )
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Robust LM WX test",
                reg.k-1,
                reg.rlm_wx[0],
                reg.rlm_wx[1],
            )
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Lagrange Multiplier (lag)",
                1,
                reg.lm_lag[0],
                reg.lm_lag[1],
            )
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Robust LM Lag - SDM",
                1,
                reg.rlm_durlag[0],
                reg.rlm_durlag[1],
            )
            strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Joint test for SDM",
                reg.k,
                reg.lm_spdurbin[0],
                reg.lm_spdurbin[1],
            )
            #strSummary += (
            #    "\n- Spatial Error and WX -\nTEST                              DF       VALUE           PROB\n"
            #)
            #strSummary += "%-27s      %2d    %12.3f        %9.4f\n\n" % (
            #    "Joint test for Error and WX",
            #    reg.k,
            #    reg.lm_slxerr[0],
            #    reg.lm_slxerr[1],
            #)

    return strSummary

def _nonspat_top(reg, ml=False):
    if not ml:
        reg.sig2ML = reg.sig2n
        reg.f_stat = diagnostics.f_stat(reg)
        reg.logll = diagnostics.log_likelihood(reg)
        reg.aic = diagnostics.akaike(reg)
        reg.schwarz = diagnostics.schwarz(reg)

        strSummary = "%-20s:%12.6g                %-22s:%12.4f\n" % (
            "Sum squared residual", reg.utu, "F-statistic", reg.f_stat[0],)
        strSummary += "%-20s:%12.3f                %-22s:%12.4g\n" % (
            "Sigma-square",
            reg.sig2,
            "Prob(F-statistic)",
            reg.f_stat[1],
        )
        strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            "S.E. of regression",
            np.sqrt(reg.sig2),
            "Log likelihood",
            reg.logll,
        )
        strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            "Sigma-square ML",
            reg.sig2ML,
            "Akaike info criterion",
            reg.aic,
        )
        strSummary += "%-20s:%12.4f                %-22s:%12.3f\n" % (
            "S.E of regression ML",
            np.sqrt(reg.sig2ML),
            "Schwarz criterion",
            reg.schwarz,
        )
    else:
        strSummary = "%-20s:%12.4f\n" % (
            "Log likelihood",
            reg.logll,
        )
        strSummary += "%-20s:%12.4f                %-22s:%12.3f\n" % (
            "Sigma-square ML",
            reg.sig2,
            "Akaike info criterion",
            reg.aic,
        )
        strSummary += "%-20s:%12.4f                %-22s:%12.3f\n" % (
            "S.E of regression",
            np.sqrt(reg.sig2),
            "Schwarz criterion",
            reg.schwarz,
        )

    return strSummary

def _nonspat_mid(reg, white_test=False):
    # compute diagnostics
    reg.mulColli = diagnostics.condition_index(reg)
    reg.jarque_bera = diagnostics.jarque_bera(reg)
    reg.breusch_pagan = diagnostics.breusch_pagan(reg)
    reg.koenker_bassett = diagnostics.koenker_bassett(reg)

    if white_test:
        reg.white = diagnostics.white(reg)

    strSummary = "\nREGRESSION DIAGNOSTICS\n"
    if reg.mulColli:
        strSummary += "MULTICOLLINEARITY CONDITION NUMBER %15.3f\n\n" % (reg.mulColli)
    strSummary += "TEST ON NORMALITY OF ERRORS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d %14.3f        %9.4f\n\n" % (
        "Jarque-Bera",
        reg.jarque_bera["df"],
        reg.jarque_bera["jb"],
        reg.jarque_bera["pvalue"],
    )
    strSummary += "DIAGNOSTICS FOR HETEROSKEDASTICITY\n"
    strSummary += "RANDOM COEFFICIENTS\n"
    strSummary += "TEST                             DF        VALUE           PROB\n"
    strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
        "Breusch-Pagan test",
        reg.breusch_pagan["df"],
        reg.breusch_pagan["bp"],
        reg.breusch_pagan["pvalue"],
    )
    strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
        "Koenker-Bassett test",
        reg.koenker_bassett["df"],
        reg.koenker_bassett["kb"],
        reg.koenker_bassett["pvalue"],
    )
    try:
        if reg.white:
            strSummary += "\nSPECIFICATION ROBUST TEST\n"
            if len(reg.white) > 3:
                strSummary += reg.white + "\n"
            else:
                strSummary += (
                    "TEST                             DF        VALUE           PROB\n"
                )
                strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                    "White",
                    reg.white["df"],
                    reg.white["wh"],
                    reg.white["pvalue"],
                )
    except:
        pass
    return strSummary

def _spat_pseudo_r2(reg):
    if np.abs(reg.rho) < 1:
        reg.pr2_e = diagnostics_tsls.pr2_spatial(reg)
        strSummary = "%-20s:  %5.4f\n" % ("Spatial Pseudo R-squared", reg.pr2_e)
    else:
        strSummary =  "Spatial Pseudo R-squared: omitted due to rho outside the boundary (-1, 1).\n"
    return strSummary

def _summary_vm(reg):
    strVM = "\n"
    strVM += "COEFFICIENTS VARIANCE MATRIX\n"
    strVM += "----------------------------\n"
    try:
        for name in reg.name_z:
            strVM += "%12s" % (name)
    except:
        for name in reg.name_x:
            strVM += "%12s" % (name)
    strVM += "\n"
    nrow = reg.vm.shape[0]
    ncol = reg.vm.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            strVM += "%12.6f" % (reg.vm[i][j])
        strVM += "\n"
    return strVM

def _summary_iteration(reg):
    """Reports the number of iterations computed and the type of estimator used for hom and het models."""
    try:
        niter = reg.niter
    except:
        niter = reg.iteration

    txt = "%-20s:%12s\n" % ("N. of iterations", niter)

    try:
        if reg.step1c:
            step1c = "Yes"
        else:
            step1c = "No"
        txt = txt[:-1] + "                %-22s:%12s\n" % (
            "Step1c computed",
            step1c,
        )
    except:
       pass

    try:
        txt = txt[:-1] + "                %-22s:%12s" % (
            "A1 type: ",
            reg.A1,
        )
    except:
        pass

    return txt

def _summary_impacts(reg, w, spat_impacts, slx_lags=0, slx_vars="All",regimes=False):
    """
    Spatial direct, indirect and total effects in spatial lag model.
    Uses multipliers computed by sputils.spmultipliers.

    Attributes
    ----------
    reg:     spreg regression object
    w:      spatial weights object
    spat_impacts:  spatial impacts method as string or list with strings
    slx_lags: int, number of spatial lags of X in the model
    slx_vars : either "All" (default) for all variables lagged, or a list
               of booleans matching the columns of x that will be lagged or not
    regimes: boolean, True if regimes model

    Returns
    -------
    sp_multipliers: dict with direct, indirect and total multipliers
    strSummary: strings with direct, indirect and total effects

    """
    try:
        spat_impacts = [spat_impacts.lower()]
    except AttributeError:
        spat_impacts = [x.lower() for x in spat_impacts]

    #variables = reg.output.query("var_type in ['x', 'yend'] and index != 0") # excludes constant
    variables = reg.output[reg.output["var_type"] == 'x']

    if regimes:
        variables = variables[~variables['var_names'].str.endswith('_CONSTANT')]
    variables_index = variables.index

    if slx_lags==0:
        strSummary = "\nSPATIAL LAG MODEL IMPACTS\n"
    else:
        strSummary = "\nSPATIAL DURBIN MODEL IMPACTS\n"

    if abs(reg.rho) >= 1:
        strSummary += "Omitted since spatial autoregressive parameter is outside the boundary (-1, 1).\n"
        return None, strSummary

    if "all" in spat_impacts:
        spat_impacts = ["simple", "full", "power"]

    sp_multipliers = {}
    for i in spat_impacts:
        spmult = spmultiplier(w, reg.rho, method=i)   # computes the multipliers, slx_lags not needed
        
        strSummary += spmult["warn"]
        btot, bdir, bind = _sp_effects(reg, variables, spmult, slx_lags,slx_vars)  # computes the impacts, needs slx_lags
        sp_multipliers[spmult["method"]] = spmult['adi'], spmult['aii'].item(), spmult['ati'].item()

        strSummary += "Impacts computed using the '" + spmult["method"] + "' method.\n"
        strSummary += "            Variable         Direct        Indirect          Total\n"
        for i in range(len(variables)):
            strSummary += "%20s   %12.4f    %12.4f    %12.4f\n" % (
            variables['var_names'][variables_index[i]], bdir[i][0], bind[i][0], btot[i][0])

    return sp_multipliers, strSummary

def _summary_vif(reg):
    """
    Summary of variance inflation factors for the model.

    Parameters
    ----------
    reg:     spreg regression object

    Returns
    -------
    strSummary: string with variance inflation factors

    """
    vif = diagnostics.vif(reg)
    strSummary = "\nVARIANCE INFLATION FACTOR\n"
    strSummary += "            Variable             VIF      Tolerance\n"
    for i in range(len(reg.name_x)-1):
        i += 1
        strSummary += "%20s    %12.4f   %12.4f\n" % (
        reg.name_x[i], vif[i][0], vif[i][1])
    return strSummary

def _summary_dwh(reg):
    """
    Summary of Durbin-Wu-Hausman test on endogeneity of variables.

    Parameters
    ----------
    reg:     spreg regression object

    Returns
    -------
    strSummary: string with Durbin-Wu-Hausman test results

    """
    strSummary = "\nREGRESSION DIAGNOSTICS\n"
    strSummary += (
            "TEST                              DF         VALUE           PROB\n")
    strSummary += "%-27s      %2d   %12.3f        %9.4f\n" % (
                "Durbin-Wu-Hausman test",reg.yend.shape[1],reg.dwh[0],reg.dwh[1])
    return strSummary

def _nslx_out(reg, section):
    """
    Summary of the NSLX model.

    Parameters
    ----------
    reg:     spreg regression object
    section: string, "top" for top part of summary, "mid" for middle part

    Returns
    -------
    strSummary: string with NSLX model summary specifics

    """
    if section == "top":
        strSummary = "%-20s:%12.3f                %-22s:%12.6g\n" % (
                "Sigma-square", reg.sign, "Sum squared residual", reg.utu,)
        strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            "S.E. of regression", np.sqrt(reg.sig2), "Log likelihood",
            reg.ll,)
        strSummary += "%-20s:%12.3f                %-22s:%12.3f\n" % (
            "Schwarz criterion", reg.schwarz, "Akaike info criterion",
            reg.aic,) 
        if len(''.join(reg.name_coords)) <= 12:       
            strSummary += "%-20s:%12s                %-22s:%12s\n" % (
                "Coordinates", reg.name_coords, "Distance metric", reg.distance_metric,)        
        else:
            max_spaces = 27
            if isinstance(reg.name_coords, list):
                name_coords = ', '.join(reg.name_coords) if len(reg.name_coords) > 1 else reg.name_coords[0]
            else:
                name_coords = reg.name_coords  # It's already a string
            if len(name_coords) > max_spaces:
                name_coords = name_coords[:max_spaces-4] + "..."
            space_between = (max_spaces - len(name_coords)) * ' '
            strSummary += "%-20s: %12s%s%-22s:%12s\n" % ("Coordinates", name_coords, space_between, "Distance metric", reg.distance_metric) 

    if section == "mid":
        strSummary = ""
        if len(reg.transform) == 1:
            strSummary += "Transformation: " + reg.transform[0] + "\n"
            strSummary += "KNN: " + str(reg.knn[0]) + "\n"
            strSummary += "Distance upper bound: " + str(reg.d_upper_bound[0]) + "\n"
        else:
            param = [reg.transform, reg.knn, reg.d_upper_bound]
            text = ["Transformation: ", "KNN: ", "Distance upper bound: "]
            for i in range(3):
                param_i = ', '.join(map(str, param[i]))     
                text_wrapper = TW.TextWrapper(width=76, subsequent_indent="             ")
                strSummary += text[i] + text_wrapper.fill(param_i) + "\n"
    return strSummary

def _probit_out(reg, spat_diag=False, sptests_only=False):
    """
    Summary of the Probit model.

    Parameters
    ----------
    reg:     spreg regression object

    Returns
    -------
    strSummary: string with Probit model summary specifics

    """

    if not sptests_only:
        strSummary_top = "%-20s:%12.2f                %-22s:%12d\n%-20s:%12.2f                %-22s:%12d\n%-20s:%12.2f                %-22s:%12.4f\n%-20s:%12.2f                %-22s:%12.4f\n%-20s:%12.2f                %-22s:%12.4f\n" % (
        "True positive rate",
        reg.fit["TPR"],
        "Number of Variables",
        reg.k,
        "True negative rate",
        reg.fit["TNR"],
        "Degrees of Freedom",
        reg.n - reg.k,
        "Balanced accuracy",
        reg.fit["BA"],
        "Log-Likelihood",
        reg.logl,            
        "% correctly pred.",
        reg.fit["PREDPC"],
        "LR test",
        reg.LR[0],
        "McFadden's rho",
        reg.mcfadrho,
        "LR test (p-value)",
        reg.LR[1],
        )

        strSummary_mid = "\nMARGINAL EFFECTS\n"
        if reg.scalem == "phimean":
            strSummary_mid += "Method: Mean of individual marginal effects\n"
        elif reg.scalem == "xmean":
            strSummary_mid += "Method: Marginal effects at variables mean\n"
        strSummary_mid += "------------------------------------------------------------------------------------\n"
        strSummary_mid += "            Variable      Slope            Std.Error     z-Statistic     Probability\n"
        strSummary_mid += "------------------------------------------------------------------------------------\n"
        for i in range(len(reg.slopes)):
            strSummary_mid += "%20s    %12.5f    %12.5f    %12.5f    %12.5f\n" % (
                reg.name_x[i + 1],
                reg.slopes[i][0],
                reg.slopes_std_err[i],
                reg.slopes_z_stat[i][0],
                reg.slopes_z_stat[i][1],
            )
        strSummary_mid += "------------------------------------------------------------------------------------\n"     
    
        strSummary_end = "\nCONFUSION MATRIX\n"
        data = {
            "Positive (1)": [
                reg.predtable["actpos"],  # Observed positives
                reg.predtable["predpos"],  # Predicted positives
                reg.predtable["truepos"],  # Correctly predicted positives
                reg.predtable["falseneg"]  # Wrongly predicted positives
            ],
            "Negative (0)": [
                reg.predtable["actneg"],  # Observed negatives
                reg.predtable["predneg"],  # Predicted negatives
                reg.predtable["trueneg"],  # Correctly predicted negatives
                reg.predtable["falsepos"]  # Wrongly predicted negatives
            ]
        }

        rows = ["Observed", "Predicted", "Correctly Predicted", "Wrongly Predicted"]
        strSummary_end += f"{'':<26}{'Positive (1)':<18}{'Negative (0)':<15}\n"
        for row, pos, neg in zip(rows, data["Positive (1)"], data["Negative (0)"]):
            strSummary_end += f"{row:<30}{pos:<15}{neg:<15}\n"
        strSummary_end += "\n------------------------------------------------------------------------------------\n"

    if spat_diag or sptests_only:
        try:
            strSummary_end += "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
        except:
            strSummary_end = "\nDIAGNOSTICS FOR SPATIAL DEPENDENCE\n"
        strSummary_end += "TEST                          DF             VALUE           PROB\n"  
        strSummary_end += "%-23s      %2d       %12.3f        %9.4f\n" % (
            "Kelejian-Prucha (error)",
            1,
            reg.KP_error[0],
            reg.KP_error[1],
        )
        strSummary_end += "%-23s      %2d       %12.3f        %9.4f\n" % (
            "Pinkse (error)",
            1,
            reg.Pinkse_error[0],
            reg.Pinkse_error[1],
        )
        strSummary_end += "%-23s      %2d       %12.3f        %9.4f\n\n" % (
            "Pinkse-Slade (error)",
            1,
            reg.PS_error[0],
            reg.PS_error[1],
        )
        if sptests_only:
            return strSummary_end
    else:
        strSummary_end = ""

    return strSummary_top, strSummary_mid, strSummary_end