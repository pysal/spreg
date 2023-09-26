.. _api_ref:

.. currentmodule:: spreg

API reference
=============

.. _models_api:

Spatial Regression Models
-------------------------

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

Discrete Choice Models
----------------------
.. autosummary::
    :toctree: generated/

    spreg.Probit

Regimes Models
---------------

Regimes models are variants of spatial regression models which allow for structural instability in parameters. That means that these models allow different coefficient values in distinct subsets of the data. 

.. autosummary::
    :toctree: generated/

    spreg.OLS_Regimes
    spreg.TSLS_Regimes
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
    spreg.Skater_reg

Seemingly-Unrelated Regressions
--------------------------------

Seemingly-unrelated regression models are a generalization of linear regression. These models (and their spatial generalizations) allow for correlation in the residual terms between groups that use the same model. In spatial Seeimingly-Unrelated Regressions, the error terms across groups are allowed to exhibit a structured type of correlation: spatial correlation. 

.. autosummary::
   :toctree: generated/
    
    spreg.SUR
    spreg.SURerrorGM
    spreg.SURerrorML
    spreg.SURlagIV
    spreg.ThreeSLS

Spatial Panel Models
--------------------

Spatial panel models allow for evaluating correlation in both spatial and time dimensions. 

.. autosummary::
   :toctree: generated/
    
    spreg.Panel_FE_Lag
    spreg.Panel_FE_Error
    spreg.Panel_RE_Lag
    spreg.Panel_RE_Error
    spreg.GM_KKP

Diagnostics
-----------

Diagnostic tests are useful for identifying model fit, sufficiency, and specification correctness. 

.. autosummary:: 
    :toctree: generated/

    spreg.f_stat
    spreg.t_stat
    spreg.r2
    spreg.ar2
    spreg.se_betas
    spreg.log_likelihood
    spreg.akaike
    spreg.schwarz
    spreg.condition_index
    spreg.jarque_bera
    spreg.breusch_pagan
    spreg.white
    spreg.koenker_bassett
    spreg.vif
    spreg.likratiotest
    spreg.LMtests
    spreg.MoranRes
    spreg.AKtest
    spreg.sur_setp
    spreg.sur_lrtest
    spreg.sur_lmtest
    spreg.lam_setp
    spreg.surLMe
    spreg.surLMlag
    spreg.constant_check
    spreg.panel_LMlag
    spreg.panel_LMerror
    spreg.panel_rLMlag
    spreg.panel_rLMerror
    spreg.panel_Hausman
