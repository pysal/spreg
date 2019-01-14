.. _api_ref:

.. currentmodule:: spreg

API reference
=============

.. _models_api:

Spatial Regression Models
=========================

These are the standard spatial regression models supported by the `spreg` package. Each of them contains a significant amount of detail in their docstring discussing how they're used, how they're fit, and how to interpret the results. 

.. autosummary::
   :toctree: generated/
    
    spreg.OLS
    spreg.ML_Lag
    spreg.ML_Error
    spreg.GM_Lag
    spreg.GM_Error
    spreg.GM_Error_Het
    spreg.GM_Error_Hom
    spreg.GM_Combo
    spreg.GM_Combo_Het
    spreg.GM_Combo_Hom
    spreg.GM_Endog_Error
    spreg.GM_Endog_Error_Het
    spreg.GM_Endog_Error_Hom
    spreg.TSLS
    spreg.ThreeSLS

Regimes Models
---------------

Regimes models are variants of spatial regression models which allow for structural instability in parameters. That means that these models allow different coefficient values in distinct subsets of the data. 

.. autosummary::
    :toctree: generated/

    spreg.OLS_Regimes
    spreg.ML_Lag_Regimes
    spreg.ML_Error_Regimes
    spreg.GM_Lag_Regimes
    spreg.GM_Error_Regimes
    spreg.GM_Error_Het_Regimes
    spreg.GM_Error_Hom_Regimes
    spreg.GM_Combo_Regimes
    spreg.GM_Combo_Hom_Regimes
    spreg.GM_Combo_Het_Regimes
    spreg.GM_Endog_Error_Regimes
    spreg.GM_Endog_Error_Hom_Regimes
    spreg.GM_Endog_Error_Het_Regimes

Seemingly-Unrelated Regressions
--------------------------------

Seeimingly-unrelated regression models are a generalization of linear regression. These models (and their spatial generalizations) allow for correlation in the residual terms between groups that use the same model. In spatial Seeimingly-Unrelated Regressions, the error terms across groups are allowed to exhibit a structured type of correlation: spatail correlation. 

.. autosummary::
   :toctree: generated/
    
    spreg.SUR
    spreg.SURerrorGM
    spreg.SURerrorML
    spreg.SURlagIV
    spreg.ThreeSLS

Diagnostics
-----------

Diagnostic tests are useful for identifying model fit, sufficiency, and specification correctness. 

.. autosummary:: 
    :toctree: generated/

    spreg.diagnostics.f_stat 
    spreg.diagnostics.t_stat 
    spreg.diagnostics.r2 
    spreg.diagnostics.ar2 
    spreg.diagnostics.se_betas 
    spreg.diagnostics.log_likelihood 
    spreg.diagnostics.akaike 
    spreg.diagnostics.schwarz
    spreg.diagnostics.condition_index 
    spreg.diagnostics.jarque_bera 
    spreg.diagnostics.breusch_pagan 
    spreg.diagnostics.white 
    spreg.diagnostics.koenker_bassett 
    spreg.diagnostics.vif 
    spreg.diagnostics.likratiotest
    spreg.diagnostics_sp.LMtests
    spreg.diagnostics_sp.MoranRes
    spreg.diagnostics_sp.AKtest
    spreg.diagnostics_sur.sur_setp
    spreg.diagnostics_sur.sur_lrtest
    spreg.diagnostics_sur.sur_lmtest
    spreg.diagnostics_sur.lam_setp
    spreg.diagnostics_sur.surLMe
    spreg.diagnostics_sur.surLMlag
