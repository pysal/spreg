import os
os.chdir("../..")

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd  # this dependency isn't required
import spreg
import formulaic
from libpysal.examples import load_example
from libpysal.weights import Kernel, fill_diagonal
from libpysal.common import RTOL


class TestFormula(unittest.TestCase):
    def setUp(self):
        # Load boston example
        self.boston = load_example("Bostonhsg")
        self.boston_df = gpd.read_file(self.boston.get_path("boston.shp"))

        # Make weights matrix and set diagonal to 0 (necessary for lag model)
        self.weights = Kernel(self.boston_df[["x", "y"]], k=50, fixed=False)
        self.weights = fill_diagonal(self.weights, 0)

    def tearDown(self):
        pass

    def test_empty_formula(self):
        with self.assertRaises(ValueError):
            model = spreg.from_formula("", self.boston_df)

    def test_empty_df(self):
        with self.assertRaises(NameError):
            model = spreg.from_formula("log(CMEDV) ~ ZN + CHAS", pd.DataFrame())

    def test_spatial_no_w(self):
        with self.assertRaises(ValueError):
            model = spreg.from_formula("log(CMEDV) ~ ZN + CHAS + &", self.boston_df)

    def test_bad_method(self):
        with self.assertRaises(ValueError):
            model = spreg.from_formula("log(CMEDV) ~ CHAS + &", self.boston_df, method="a")

    def test_debug(self):
        formula = "log(CMEDV) ~ ZN + RM"
        outputs = spreg.from_formula(formula, self.boston_df, debug=True)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], spreg.from_formula(formula, self.boston_df))

    def test_kwargs(self):
        # OLS formula
        formula = "log(CMEDV) ~ ZN + RM"
        model = spreg.from_formula(formula, self.boston_df, robust="white")
        self.assertEqual(model.robust, "white")
        
        # Lag formula
        formula = "log(CMEDV) ~ ZN + RM + <log(CMEDV)>"
        model = spreg.from_formula(formula, self.boston_df, sig2n_k=True)
        self.assertEqual(model.sig2n_k, True)

        # Error formula
        formula = "log(CMEDV) ~ ZN + RM + &"
        model = spreg.from_formula(formula, self.boston_df, vm=True)
        self.assertEqual(model.vm, True)

        # Combo formula
        formula = "log(CMEDV) ~ ZN + RM + <log(CMEDV)> + &"
        model = spreg.from_formula(formula, self.boston_df, vm=True)
        self.assertEqual(model.vm, True)

    def test_nonspatial_formula(self):
        formula = "log(CMEDV) ~ ZN + AGE + RM"
        model = spreg.from_formula(formula, self.boston_df)
        self.assertEqual(type(model), spreg.OLS)
        # TODO add numerical checks

    def test_nonspatial_formula_with_transforms(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2}"
        model = spreg.from_formula(formula, self.boston_df)
        self.assertEqual(type(model), spreg.OLS)
        # TODO add numerical checks

    def test_field_not_found(self):
        with self.assertRaises(KeyError):
            model = spreg.from_formula("log(CMEDV) ~ ZN + HI + INDUS")

    def test_spatial_lag_x(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <CRIM>"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.OLS)
        # TODO add numerical checks

    def test_spatial_lag_y(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <log(CMEDV)>"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Lag)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.ML_Lag)
        # TODO add numerical checks

    def test_spatial_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + &"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Error)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.ML_Error)
        # TODO add numerical checks

    def test_slx_sly(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <INDUS + log(CMEDV)>"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Lag)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.ML_Lag)
        # TODO add numerical checks

    def test_slx_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <INDUS> + &"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Error)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.ML_Error)
        # TODO add numerical checks

    def test_sly_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <log(CMEDV)> + &"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Combo)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.GM_Combo)

    def test_slx_sly_error(self):
        formula = "log(CMEDV) ~ ZN + AGE + {RM**2} + <INDUS + log(CMEDV)> + &"
        model = spreg.from_formula(formula, self.boston_df, w=self.weights)
        self.assertEqual(type(model), spreg.GM_Combo)
        # TODO add numerical checks

        model = spreg.from_formula(formula, self.boston_df, w=self.weights, method="lu")
        self.assertEqual(type(model), spreg.GM_Combo)

    def test_pathological_formulae(self):
        with self.assertRaises(formulaic.errors.FormulaSyntaxError):
            model = spreg.from_formula("log(CMEDV) ~ ZN + & + &", self.boston_df,
                                       w=self.weights)
    
        with self.assertRaises(formulaic.errors.FormulaSyntaxError):
            model = spreg.from_formula("log(CMEDV) ~ <ZN + <AGE>>", self.boston_df,
                                       w=self.weights)

if __name__ == "__main__":
    unittest.main()
