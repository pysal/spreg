"""
spsearch.py: specification search strategies following Anselin, Serenini, Amaral (2024)

"""

__author__ = "Luc Anselin lanselin@gmail.com,\
     Pedro Amaral pedrovma@gmail.com,\
     Renan Serenini renan.serenini@uniroma1.it"

import numpy as np
from . import ols as OLS
from . import twosls_sp as STSLS
from .diagnostics_sp import LMtests, AKtest
from . import error_sp as ERROR


__all__ = ["stge_classic", "stge_kb", "stge_pre", "gets_gns", "gets_sdm"]


def stge_classic(
    y,
    x,
    w,
    w_lags=2,
    robust=None,
    sig2n_k=True,
    name_y=False,
    name_x=False,
    name_w=False,
    name_ds=False,
    latex=False,
    p_value=0.01,
    finmod=True,
    mprint=True,
):
    """
    Classic forward specification: Evaluate results from LM-tests and their robust versions from spreg.OLS.
    Estimate lag model with AK test if warranted.

    Arguments:
    ----------
    y        : dependent variable
    x        : matrix of explanatory variables
    w        : spatial weights
    w_lags   : number of lags to be used as instruments in S2SLS
    robust   : White standard errors?
    sig2n_k  : error variance estimate (consistent or unbiased=True)
    name_y   : name of dependent variable (string)
    name_x   : list of strings with x-variable names
    name_w   : string with name for spatial weights
    name_ds  : string with name for data set
    latex    : flag for latex output
    p_value  : significance threshold
    finmod   : flag for estimation of final model
    mprint   : flag for regression summary as search result

    Returns:
    ----------
    result: the selected model as a string
            0 = OLS
            1 = LAG
            2 = ERROR
            3 = LAGr
            4 = ERRORr
            5 = LAG_Br
            6 = ERROR_Br
            7 = LAG_Nr
            8 = ERROR_Nr
            9 = SARSAR
    finreg: regression object for final model

    Example:
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import stge_classic

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    >>> y_var = "CRIME"
    >>> y = np.array([db.by_col(y_var)]).reshape(49, 1)
    >>> x_var = ["INC", "HOVAL"]
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = "r"
    >>> name_y = y_var
    >>> name_x = x_var
    >>> name_w = "Rook Weights"
    >>> name_ds = "Columbus Data"

    >>> result, finreg = stge_classic(y, x, w, mprint=True,
    ...                               name_y=name_y, name_x=name_x, name_w=name_w, name_ds=name_ds)
    Model selected by STGE-Classic: LAG
    REGRESSION RESULTS
    ------------------
    <BLANKLINE>
    SUMMARY OF OUTPUT: SPATIAL TWO STAGE LEAST SQUARES
    --------------------------------------------------
    Data set            :Columbus Data
    Weights matrix      :Rook Weights
    Dependent Variable  :       CRIME                Number of Observations:          49
    Mean dependent var  :     35.1288                Number of Variables   :           4
    S.D. dependent var  :     16.7321                Degrees of Freedom    :          45
    Pseudo R-squared    :      0.6513
    Spatial Pseudo R-squared:  0.5733
    <BLANKLINE>
    ------------------------------------------------------------------------------------
                Variable     Coefficient       Std.Error     z-Statistic     Probability
    ------------------------------------------------------------------------------------
                CONSTANT        45.45909        10.72499         4.23861         0.00002
                     INC        -1.04101         0.37241        -2.79531         0.00519
                   HOVAL        -0.25954         0.08855        -2.93085         0.00338
                 W_CRIME         0.41929         0.17977         2.33245         0.01968
    ------------------------------------------------------------------------------------
    Instrumented: W_CRIME
    Instruments: W2_HOVAL, W2_INC, W_HOVAL, W_INC
    <BLANKLINE>
    DIAGNOSTICS FOR SPATIAL DEPENDENCE
    TEST                              DF         VALUE           PROB
    Anselin-Kelejian Test             1          0.130           0.7185
    <BLANKLINE>
    SPATIAL LAG MODEL IMPACTS
                Variable         Direct        Indirect          Total
                     INC        -1.0410         -0.7517         -1.7927
                   HOVAL        -0.2595         -0.1874         -0.4469
    ================================ END OF REPORT =====================================


    """

    finreg = False
    p = p_value
    k = 0  # indicator for type of final model 0 = OLS; 1 = Lag; 2 = Error; 3 = SAR-SAR; 4 = LAG from SAR

    if not (name_y) or not (name_x):
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

        name_y = model_ols_1.name_y
        name_x = model_ols_1.name_x[1:]

    else:
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_y=name_y,
            name_x=name_x,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

    pvals = [
        model_ols_1.lm_error[1],
        model_ols_1.lm_lag[1],
        model_ols_1.rlm_error[1],
        model_ols_1.rlm_lag[1],
        model_ols_1.lm_sarma[1],
    ]

    p_error, p_lag, p_rerror, p_rlag, p_sarma = pvals
    if p_lag >= p and p_error >= p:  # First test, no LM significant= Stop and keep OLS
        result = "OLS"
        k = 0
    else:
        # Just one significant
        if p_lag < p and p_error >= p:
            result = "LAG"
            k = 1
        elif p_lag >= p and p_error < p:
            result = "ERROR"
            k = 2
        # Both are significant (Check robust version)
        elif p_lag < p and p_error < p:
            # One robust significant
            if p_rlag < p and p_rerror >= p:
                result = "LAGr"
                k = 1
            elif p_rlag >= p and p_rerror < p:
                result = "ERRORr"
                k = 2
            # Both robust are significant (look for the most significant)
            elif p_rlag < p and p_rerror < p:
                # check AK in lag model
                try:
                    model_lag = STSLS.GM_Lag(
                        y,
                        x,
                        w=w,
                        slx_lags=0,
                        w_lags=w_lags,
                        hard_bound=True,
                        name_y=name_y,
                        name_x=name_x,
                        name_w=name_w,
                        name_ds=name_ds,
                        latex=latex,
                    )

                    ak_lag = AKtest(model_lag, w, case="gen")
                    if ak_lag.p <= p:
                        result = "SARSAR"
                        k = 3
                    elif p_rlag <= p_rerror:
                        result = "LAG_BR"
                        k = 4
                    elif p_rlag > p_rerror:
                        result = "ERROR_Br"
                        k = 2
                except:
                    if p_rlag <= p_rerror:
                        result = "LAG_BR"
                        k = 1
                    else:
                        result = "ERROR_Br"
                        k = 2

            else:  # None robust are significant (still look for the 'most significant')
                if p_rlag <= p_rerror:
                    result = "LAG_Nr"
                    k = 4
                elif p_rlag > p_rerror:
                    result = "ERROR_Nr"
                    k = 2

    if finmod:  # pass final regression
        msel = "Model selected by STGE-Classic: "
        if k == 0:  # OLS
            finreg = model_ols_1

        elif (k == 1) or (k == 4):  # LAG
            try:
                finreg = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GM LAG parameters outside bounds"
                finreg = False
        elif k == 2:  # ERROR
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GMM Error parameters outside bounds"
                finreg = False
        elif k == 3:  # SARSAR
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    add_wy=True,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: SARSAR parameters outside bounds"
                finreg = False

        #        elif k == 4: # LAG already computed
        #            finreg = model_lag
        if mprint:
            print(msel + result)
            if not (finreg == False):  # cannot print when finreg=False
                print(finreg.summary)

    return (result, finreg)


def stge_kb(
    y,
    x,
    w,
    w_lags=2,
    robust=None,
    sig2n_k=True,
    name_y=False,
    name_x=False,
    name_w=False,
    name_ds=False,
    latex=False,
    p_value=0.01,
    finmod=True,
    mprint=True,
):
    """
    Forward specification: Evaluate results from Koley-Bera LM-tests and their robust versions from spreg.OLS.

    Arguments:
    ----------
    y        : dependent variable
    x        : matrix of explanatory variables
    w        : spatial weights
    w_lags   : number of lags to be used as instruments in S2SLS
    robust   : White standard errors?
    sig2n_k  : error variance estimate (consistent or unbiased=True)
    name_y   : name of dependent variable (string)
    name_x   : list of strings with x-variable names
    name_w   : string with name for spatial weights
    name_ds  : string with name for data set
    latex    : flag for latex output
    p_value  : significance threshold
    finmod   : flag for estimation of final model
    mprint   : flag for regression summary as search result

    Returns:
    ----------
    result: the selected model as a string
            0 = OLS
            1 = LAG
            2 = SLX
            3 = SDM
    finreg: regression object for final model

    Example:
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import stge_kb

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    >>> y_var = "CRIME"
    >>> y = np.array([db.by_col(y_var)]).reshape(49, 1)
    >>> x_var = ["INC", "HOVAL"]
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = "r"
    >>> name_y = y_var
    >>> name_x = x_var
    >>> name_w = "Rook Weights"
    >>> name_ds = "Columbus Data"

    >>> result, finreg = stge_kb(y, x, w, name_y=name_y, name_x=name_x,
    ...                               name_w=name_w, name_ds=name_ds, mprint=False)
    >>> print("Model selected by STGE-KB:",result)
    Model selected by STGE-KB: OLS


    """

    finreg = False
    p = p_value
    k = 0  # indicator for type of final model 0 = OLS; 1 = Lag; 2 = SLX; 3 = SDM

    if not (name_y) or not (name_x):
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

        name_y = model_ols_1.name_y
        name_x = model_ols_1.name_x[1:]

    else:
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_y=name_y,
            name_x=name_x,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

    pvals = [
        model_ols_1.rlm_wx[1],
        model_ols_1.rlm_durlag[1],
        model_ols_1.lm_spdurbin[1],
    ]

    p_rlwx, p_rdury, p_spdur = pvals

    # first check following KB(2024) - joint test on SDM
    if p_spdur > p:  # not significant
        result = "OLS"
        k = 0
    else:  # joint test is significant
        if p_rlwx < p and p_rdury < p:
            result = "SDM"
            k = 3
        elif p_rdury < p:  # only robust lag
            result = "LAG"
            k = 1
        elif p_rlwx < p:  # only robust WX
            result = "SLX"
            k = 2
        else:  # should never be reached
            result = "OLS"
            k = 0

    if finmod:  # pass final regression
        msel = "Model selected by STGE-KB: "
        if k == 0:  # OLS
            finreg = model_ols_1
        elif k == 1:  # LAG
            try:
                finreg = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GM Lag parameters outside bounds"
                finreg = False
        elif k == 2:  # SLX
            finreg = OLS.OLS(
                y,
                x,
                w=w,
                slx_lags=1,
                spat_diag=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        elif k == 3:  # SDM
            try:
                finreg = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=1,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: SDM parameters outside bounds"
                finreg = False
        if mprint:
            print(msel + result)
            if not (finreg == False):  # cannot print when finreg=False
                print(finreg.summary)

    return (result, finreg)


def stge_pre(
    y,
    x,
    w,
    w_lags=2,
    robust=None,
    sig2n_k=True,
    name_y=False,
    name_x=False,
    name_w=False,
    name_ds=False,
    latex=False,
    p_value=0.01,
    finmod=True,
    mprint=True,
):
    """
    Forward specification: Evaluate results from Koley-Bera LM-tests to decide on OLS vs SLX then
    proceed as in stge_classic.

    Arguments:
    ----------
    y        : dependent variable
    x        : matrix of explanatory variables
    w        : spatial weights
    w_lags   : number of lags to be used as instruments in S2SLS
    robust   : White standard errors?
    sig2n_k  : error variance estimate (consistent or unbiased=True)
    name_y   : name of dependent variable (string)
    name_x   : list of strings with x-variable names
    name_w   : string with name for spatial weights
    name_ds  : string with name for data set
    latex    : flag for latex output
    p_value  : significance threshold
    finmod   : flag for estimation of final model
    mprint   : flag for regression summary as search result

    Returns:
    ----------
    result: the selected model as a string
            0 = OLS
            1 = LAG
            2 = ERROR
            3 = SARMA
            4 = SLX
            5 = SDM
            6 = SLX-ERR
            7 = GNS
    finreg: regression object for final model

    Example:
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import stge_pre

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    >>> y_var = "CRIME"
    >>> y = np.array([db.by_col(y_var)]).reshape(49, 1)
    >>> x_var = ["INC", "HOVAL"]
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = "r"
    >>> name_y = y_var
    >>> name_x = x_var
    >>> name_w = "Rook Weights"
    >>> name_ds = "Columbus Data"

    >>> result, finreg = stge_pre(y, x, w, name_y=name_y, name_x=name_x,
    ...                               name_w=name_w, name_ds=name_ds, mprint=False)
    >>> print("Model selected by STGE-Pre:",result)
    Model selected by STGE-Pre: LAG

    """

    finreg = False
    p = p_value
    k = 0  # indicator for type of final model 0 = OLS; 1 = Lag; 2 = Error; 3 = SARSAR;
    # 4 = SLX; 5 = SDM; 6 = SLX-Err; 7 = GNS

    models = ["OLS", "LAG", "ERROR", "SARSAR", "SLX", "SDM", "SLX-ERR", "GNS"]

    if not (name_y) or not (name_x):
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

        name_y = model_ols_1.name_y
        name_x = model_ols_1.name_x[1:]

    else:
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=0,
            spat_diag=True,
            name_y=name_y,
            name_x=name_x,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

    pv1 = model_ols_1.lm_wx[1]  # LM test on WX
    pv2 = model_ols_1.rlm_wx[1]  # robust LM test in presence of rho

    # selection of OLS or SLX
    if pv1 < p and pv2 < p:  # proceed with SLX results
        slx = 1
        model_ols_1 = OLS.OLS(
            y,
            x,
            w=w,
            slx_lags=1,
            spat_diag=True,
            name_y=name_y,
            name_x=name_x,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )
    else:  # stay with OLS estimation
        slx = 0

    idv = slx * 4  # keeps track of model

    pvals = [
        model_ols_1.lm_error[1],
        model_ols_1.lm_lag[1],
        model_ols_1.rlm_error[1],
        model_ols_1.rlm_lag[1],
        model_ols_1.lm_sarma[1],
    ]

    p_error, p_lag, p_rerror, p_rlag, p_sarma = pvals

    if (
        p_lag >= p and p_error >= p
    ):  # First test, no LM significant= Stop and keep OLS or SLX
        k = idv + 0
        result = models[k]
    else:
        # Just one significant
        if p_lag < p and p_error >= p:
            k = idv + 1
            result = models[k]

        elif p_lag >= p and p_error < p:
            k = idv + 2
            result = models[k]

        # Both are significant (Check robust version)
        elif p_lag < p and p_error < p:
            # One robust significant
            if p_rlag < p and p_rerror >= p:
                k = idv + 1
                result = models[k]

            elif p_rlag >= p and p_rerror < p:
                k = idv + 2
                result = models[k]

            # Both robust are significant (look for the most significant)
            elif p_rlag < p and p_rerror < p:
                # check AK in lag model
                try:
                    model_lag = STSLS.GM_Lag(
                        y,
                        x,
                        w=w,
                        slx_lags=slx,
                        w_lags=w_lags,
                        hard_bound=True,
                        name_y=name_y,
                        name_x=name_x,
                        name_w=name_w,
                        name_ds=name_ds,
                        latex=latex,
                    )

                    ak_lag = AKtest(model_lag, w, case="gen")
                    if ak_lag.p <= p:
                        k = idv + 3
                        result = models[k]

                    elif p_rlag <= p_rerror:
                        k = idv + 1
                        result = models[k]

                    elif p_rlag > p_rerror:
                        k = idv + 2
                        result = models[k]

                except:  # ignore lag model
                    if p_rlag <= p_rerror:
                        k = idv + 1
                        result = models[1]

                    else:
                        k = idv + 2
                        result = models[k]

            else:  # None robust are significant (still look for the 'most significant')
                if p_rlag <= p_rerror:
                    k = idv + 1
                    result = models[k]

                elif p_rlag > p_rerror:
                    k = idv + 2
                    result = models[k]

    if finmod:  # pass final regression
        msel = "Model selected by STGE-Pre: "
        if k == 0 or k == 4:
            finreg = model_ols_1
        elif k == 1 or k == 5:
            try:
                finreg = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=slx,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GM LAG parameters outside bounds"
                finreg = False
        elif k == 2 or k == 6:
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=slx,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GMM Error parameters outside bounds"
                finreg = False
        elif k == 3 or k == 7:
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=slx,
                    add_wy=True,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = (
                    result + " -- Exception: autoregressive parameters outside bounds"
                )
                finreg = False
        if mprint:
            print(msel + result)
            if not (finreg == False):  # cannot print when finreg=False
                print(finreg.summary)

    return (result, finreg)


def gets_gns(
    y,
    x,
    w,
    w_lags=2,
    robust=None,
    sig2n_k=True,
    name_y=False,
    name_x=False,
    name_w=False,
    name_ds=False,
    latex=False,
    p_value=0.01,
    finmod=True,
    mprint=True,
):
    """
    GETS specification starting with GNS model estimation. Estimate simplified model when t-tests are
    not significant.

    Arguments:
    ----------
    y        : dependent variable
    x        : matrix of explanatory variables
    w        : spatial weights
    w_lags   : number of lags to be used as instruments in S2SLS
    robust   : White standard errors?
    sig2n_k  : error variance estimate (consistent or unbiased=True)
    name_y   : name of dependent variable (string)
    name_x   : list of strings with x-variable names
    name_w   : string with name for spatial weights
    name_ds  : string with name for data set
    latex    : flag for latex output
    p_value  : significance threshold
    finmod   : flag for estimation of final model
    mprint   : flag for regression summary as search result

    Returns:
    ----------
    result: the selected model as a string
            0 = OLS
            1 = LAG
            2 = ERROR
            3 = SARSAR
            4 = SLX
            5 = SDM
            6 = SLX-Err
            7 = GNS
    finreg: regression object for final model

    Example:
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import gets_gns

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    >>> y_var = "CRIME"
    >>> y = np.array([db.by_col(y_var)]).reshape(49, 1)
    >>> x_var = ["INC", "HOVAL"]
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = "r"
    >>> name_y = y_var
    >>> name_x = x_var
    >>> name_w = "Rook Weights"
    >>> name_ds = "Columbus Data"

    >>> result, finreg = gets_gns(y, x, w, name_y=name_y, name_x=name_x,
    ...                               name_w=name_w, name_ds=name_ds, mprint=False)
    >>> print("Model selected by GETS-GNS:",result)
    Model selected by GETS-GNS: OLS

    """

    finreg = False
    p = p_value

    k = x.shape[1]

    if not (name_y) or not (name_x):
        try:
            model_gns = ERROR.GMM_Error(
                y,
                x,
                w=w,
                slx_lags=1,
                add_wy=True,
                w_lags=w_lags,
                hard_bound=True,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        except:
            result = "Exception: GNS parameters out of bounds"
            print(result)
            return (result, finreg)

        name_y = model_gns.name_y
        name_x = model_gns.name_x[1 : k + 1]

    else:
        try:
            model_gns = ERROR.GMM_Error(
                y,
                x,
                w=w,
                slx_lags=1,
                add_wy=True,
                w_lags=w_lags,
                hard_bound=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        except:
            result = "Exception: GNS parameters out of bounds"
            print(result)
            return (result, finreg)

    pstats = np.array(model_gns.z_stat)[1 + k :, 1]  # t statistics p-values
    pk = len(
        pstats
    )  # number of p-values, last one is p_lam, next to last p_rho, before that p_gam

    if pstats.max() < p:  # least significant of three is still significant
        result = "GNS"
        if finmod:
            finreg = model_gns

    elif pstats.min() >= p:  # all non-significant
        result = "OLS"

    else:  # at least one non-significant and one sig spatial parameter
        # since max is not sig, but (at least) min is
        cand = pstats.argmax()  # least significant is not sig since max > p
        if cand == (pk - 1):  # lambda not significant, but at least one of rho/gamma is
            # go to spatial Durbin - only rho and gam
            try:
                model_sdm = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=1,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = "Exception: SDM parameters out of bounds"
                print(result)
                return (result, finreg)

            pstats = np.array(model_sdm.z_stat)[1 + k :, 1]
            pk = len(pstats)
            if (
                pstats.max() < p
            ):  # least significant of two is still significant - SDM candidate
                # check on spatial common factor
                if model_sdm.cfh_test[1] < p:  # rejected - SDM
                    result = "SDM"
                    if finmod:
                        finreg = model_sdm
                else:  # not reject common factor hypothesis - ERROR
                    result = "ERROR"

            elif pstats.min() >= p:  # none significant, even bother?
                result = "OLS"

            else:  # one significant and one non-sign spatial parameter
                cand = pstats.argmax()  # non-significant one
                if cand == (pk - 1):  # rho not sig
                    result = "SLX"

                else:  # gamma not sig
                    result = "LAG"

        elif cand == (
            pk - 2
        ):  # rho not significant, but at least one of lambda/gamma is
            # go to SLX-Error
            try:
                model_slxerr = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=1,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = "Exception: SLX Error parameters out of bounds"
                print(result)
                return (result, finreg)

            pstats = np.array(model_slxerr.z_stat)[1 + k :, 1]
            pk = len(pstats)

            if pstats.max() < p:  # least significant of two is still significant
                result = "SLX-ERR"
                if finmod:
                    finreg = model_slxerr

            elif pstats.min() >= p:  # none significant, even bother?
                result = "OLS"

            else:  # one significant and one non-sign spatial parameter
                cand = pstats.argmax()  # non-significant one
                if cand == (pk - 1):  # lambda not sig
                    result = "SLX"

                else:  # gamma not sig
                    result = "ERROR"

        else:  # gamma not sig, but at least one of rho/lambda is
            # go to SARSAR
            try:
                model_sarsar = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    add_wy=True,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = "Exception: SARSAR parameters out of bounds"
                print(result)
                return (result, finreg)

            pstats = np.array(model_sarsar.z_stat)[1 + k :, 1]
            pk = len(pstats)
            if pstats.max() < p:  # least significant of two is still significant
                result = "SARSAR"
                if finmod:
                    finreg = model_sarsar
            elif pstats.min() >= p:  # none significant, even bother?
                result = "OLS"

            else:  # one significant and one non-sign spatial parameter
                cand = pstats.argmax()  # non-significant one
                if cand == (pk - 1):  # lambda not sig
                    result = "LAG"

                else:  # rho not sig
                    result = "ERROR"

    if finmod:  # pass final regression
        msel = "Model selected by GETS-GNS: "
        if result == "OLS":
            finreg = OLS.OLS(
                y,
                x,
                w=w,
                slx_lags=0,
                spat_diag=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        elif result == "SLX":
            finreg = OLS.OLS(
                y,
                x,
                w=w,
                slx_lags=1,
                spat_diag=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        elif result == "ERROR":
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GMM Error parameters outside bounds"
                finreg = False
        elif result == "LAG":
            try:
                finreg = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GM LAG parameters outside bounds"
                finreg = False
        if mprint:
            print(msel + result)
            if not (finreg == False):  # cannot print when finreg=False
                print(finreg.summary)

    return (result, finreg)


def gets_sdm(
    y,
    x,
    w,
    w_lags=2,
    robust=None,
    sig2n_k=True,
    name_y=False,
    name_x=False,
    name_w=False,
    name_ds=False,
    latex=False,
    p_value=0.01,
    finmod=True,
    mprint=True,
):
    """
    Hybrid specification search: Starting from the estimation of the Spatial Durbin model,
                          it tests significance of coefficients and carries out specification
                          tests for error autocorrelation to suggest the most appropriate model

    Arguments:
    ----------
    y        : dependent variable
    x        : matrix of explanatory variables
    w        : spatial weights
    w_lags   : number of lags to be used as instruments in S2SLS
    robust   : White standard errors?
    sig2n_k  : error variance estimate (consistent or unbiased=True)
    name_y   : name of dependent variable (string)
    name_x   : list of strings with x-variable names
    name_w   : string with name for spatial weights
    name_ds  : string with name for data set
    latex    : flag for latex output
    p_value  : significance threshold
    finmod   : flag for estimation of final model
    mprint   : flag for regression summary as search result

    Returns:
    ----------
    result: the selected model as a string
            0 = OLS
            1 = LAG
            2 = ERROR
            3 = SARSAR
            4 = SLX
            5 = SDM
            6 = SLX-Err
            7 = GNS
    finreg: regression object for final model

    Example:
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from spreg import gets_sdm

    >>> db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    >>> y_var = "CRIME"
    >>> y = np.array([db.by_col(y_var)]).reshape(49, 1)
    >>> x_var = ["INC", "HOVAL"]
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    >>> w.transform = "r"
    >>> name_y = y_var
    >>> name_x = x_var
    >>> name_w = "Rook Weights"
    >>> name_ds = "Columbus Data"

    >>> result, finreg = gets_sdm(y, x, w, name_y=name_y, name_x=name_x,
    ...                               name_w=name_w, name_ds=name_ds, mprint=False)
    >>> print("Model selected by GETS-SDM:",result)
    Model selected by GETS-SDM: OLS

    """

    finreg = False
    p = p_value

    k = x.shape[1]

    if not (name_y) or not (name_x):
        try:
            model_sdm = STSLS.GM_Lag(
                y,
                x,
                w=w,
                slx_lags=1,
                w_lags=w_lags,
                hard_bound=True,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        except:
            result = "Exception: SDM parameters out of bounds"
            print(result)
            return (result, finreg)

        name_y = model_sdm.name_y
        name_x = model_sdm.name_x[1 : k + 1]

    else:
        try:
            model_sdm = STSLS.GM_Lag(
                y,
                x,
                w=w,
                slx_lags=1,
                w_lags=w_lags,
                hard_bound=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )
        except:
            result = "Exception: SDM parameters out of bounds"
            print(result)
            return (result, finreg)

    pstats = np.array(model_sdm.z_stat)[1 + k :, 1]  # spatial parameters
    pk = len(pstats)

    if pstats.max() < p:  # least significant of two is still significant = SDM or GNS
        # check on spatial common factor
        if model_sdm.cfh_test[1] >= p:  # not rejected - ERROR
            result = "ERROR"

        else:  # could be GNS
            ak_sdm = AKtest(model_sdm, w, case="gen")
            if ak_sdm.p < p:  # remaining error
                result = "GNS"

            else:
                result = "SDM"
                if finmod:
                    finreg = model_sdm

    elif pstats.min() >= p:  # none significant - OLS or SEM
        model_ols = OLS.OLS(
            y,
            x,
            w=w,
            spat_diag=True,
            name_y=name_y,
            name_x=name_x,
            name_w=name_w,
            name_ds=name_ds,
            latex=latex,
        )

        # check on LM-Error
        errtest = LMtests(model_ols, w)
        if errtest.lme[1] < p:  # ERROR
            result = "ERROR"

        else:
            result = "OLS"
            if finmod:
                finreg = model_ols

    else:  # one significant and one non-sign spatial parameter
        cand = pstats.argmax()  # non-significant one
        if cand == (pk - 1):  # rho not sig, SLX model
            # check error in SLX
            model_slx = OLS.OLS(
                y,
                x,
                w=w,
                slx_lags=1,
                spat_diag=True,
                name_y=name_y,
                name_x=name_x,
                name_w=name_w,
                name_ds=name_ds,
                latex=latex,
            )

            errtest = LMtests(model_slx, w)
            if errtest.lme[1] < p:  # SLX-ERROR
                result = "SLX-Err"

            else:
                result = "SLX"
                if finmod:
                    finreg = model_slx
        else:  # gamma not sign, lag model
            try:
                model_lag = STSLS.GM_Lag(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = "Exception: LAG parameters out of bounds"
                print(result)
                return (result, finreg)
            # print(model_lag.summary)
            ak_lag = AKtest(model_lag, w, case="gen")
            if ak_lag.p < p:  # remaining error
                result = "SARSAR"

            else:  # no error
                result = "LAG"
                if finmod:
                    finreg = model_lag

    if finmod:  # pass final regression
        msel = "Model selected by GETS-SDM: "
        if result == "ERROR":
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=0,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GMM Error parameters outside bounds"
                finreg = False
        elif result == "SARSAR":
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    add_wy=True,
                    slx_lags=0,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: SARSAR parameters out of bounds"
                return (result, finreg)
        elif result == "SLX-Err":
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=1,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: SLX Error parameters out of bounds"
                return (result, finreg)
        elif result == "GNS":
            try:
                finreg = ERROR.GMM_Error(
                    y,
                    x,
                    w=w,
                    slx_lags=1,
                    add_wy=True,
                    w_lags=w_lags,
                    hard_bound=True,
                    name_y=name_y,
                    name_x=name_x,
                    name_w=name_w,
                    name_ds=name_ds,
                    latex=latex,
                )
            except:
                result = result + " -- Exception: GNS parameters out of bounds"
        if mprint:
            print(msel + result)
            if not (finreg == False):  # cannot print when finreg=False
                print(finreg.summary)

    return (result, finreg)


def _test():
    import doctest

    # the following line could be used to define an alternative to the '<BLANKLINE>' flag
    # doctest.BLANKLINE_MARKER = 'something better than <BLANKLINE>'
    start_suppress = np.get_printoptions()["suppress"]
    np.set_printoptions(suppress=True)
    doctest.testmod()
    np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    _test()

    import numpy as np
    import libpysal
    from spreg import stge_classic

    # Load data
    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")

    # Define the dependent variable
    y_var = "CRIME"
    y = np.array([db.by_col(y_var)]).reshape(49, 1)

    # Define the explanatory variables
    x_var = ["INC", "HOVAL"]
    x = np.array([db.by_col(name) for name in x_var]).T

    # Define the spatial weights
    w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"

    # Define the names for the variables and the dataset
    name_y = y_var
    name_x = x_var
    name_w = "Rook Weights"
    name_ds = "Columbus Data"

    # Call the stge_classic function and output the results
    result, finreg = stge_classic(
        y,
        x,
        w,
        mprint=True,
        name_y=name_y,
        name_x=name_x,
        name_w=name_w,
        name_ds=name_ds,
    )
