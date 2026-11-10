import contextlib
import io
import unittest
import libpysal
import numpy as np
import geopandas as gpd
from spreg.diagnostics_panel import panel_LMlag, panel_LMerror, panel_rLMlag
from spreg.diagnostics_panel import panel_rLMerror, panel_Hausman, BSK_tests
from spreg.panel_fe import Panel_FE_Lag
from spreg.panel_re import Panel_RE_Lag
from spreg.panel import ML_LagFE, ML_LagRE, PooledOLS, PanelFE, PanelRE
from libpysal.common import RTOL

class Test_Panel_Diagnostics(unittest.TestCase):
    def setUp(self):
        self.ds_name = "NCOVR"
        libpysal.examples.load_example(self.ds_name)

        db = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
        filter_states = ["Kansas", "Missouri", "Oklahoma", "Arkansas"]
        db = db[db["STATE_NAME"].isin(filter_states)].copy()

        self.y = db[["HR70", "HR80", "HR90"]]
        self.x = db[["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]]

        self.w = libpysal.weights.Queen.from_dataframe(db, use_index=True)
        self.w.transform = "r"

    def test_LM(self):
        lmlag = panel_LMlag(self.y, self.x, self.w)
        exp = np.array([52.06732964449592, 5.36298288480122e-13])
        np.testing.assert_allclose(lmlag, exp, RTOL)
        lmerror = panel_LMerror(self.y, self.x, self.w)
        exp = np.array([81.69630396101608, 1.5868998506678388e-19])
        np.testing.assert_allclose(lmerror, exp, RTOL)
        rlmlag = panel_rLMlag(self.y, self.x, self.w)
        exp = np.array([2.5125780962741793, 0.11294102977710921])
        np.testing.assert_allclose(rlmlag, exp, RTOL)
        rlmerror = panel_rLMerror(self.y, self.x, self.w)
        exp = np.array([32.14155241279442, 1.4333858484607395e-08])
        np.testing.assert_allclose(rlmerror, exp, RTOL)

    def test_Hausman(self):
        fe_lag = Panel_FE_Lag(self.y, self.x, self.w)
#        fe_error = Panel_FE_Error(self.y, self.x, self.w)
        re_lag = Panel_RE_Lag(self.y, self.x, self.w)
#        re_error = Panel_RE_Error(self.y, self.x, self.w)
        Hlag = panel_Hausman(fe_lag, re_lag)
        exp = np.array([67.26822586935438, 1.6407966665540482e-14])
        np.testing.assert_allclose(Hlag, exp, RTOL)
#        Herror = panel_Hausman(fe_error, re_error)
#        exp = np.array([-84.38351088621853, 1.0])
#        np.testing.assert_allclose(Herror, exp, RTOL)

    def test_Hausman_ML_lag(self):
        fe_lag = ML_LagFE(self.y, self.x, self.w)
        re_lag = ML_LagRE(self.y, self.x, self.w)
        Hlag = panel_Hausman(fe_lag, re_lag)
        exp = np.array([67.26822703188164, 1.640790543222799e-14])
        np.testing.assert_allclose(Hlag, exp, RTOL)


class Test_BSK(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.w = libpysal.weights.lat2W(5, 5)
        self.w.transform = "r"
        n, t = self.w.n, 4
        mu = np.tile(rng.normal(size=n), t)
        self.x = rng.normal(size=(n * t, 2))
        self.y = (1 + self.x @ np.array([1.0, -1.0]) + mu
                  + rng.normal(size=n * t)).reshape(-1, 1)
        self.pooled = PooledOLS(self.y, self.x, self.w, spat_diag=False)

    def test_marginal_tests_from_other_models(self):
        # LM1, LM2 and LMJ only depend on the pooled OLS residuals
        which = ["LM1", "LM2", "LMJ"]
        exp = BSK_tests(self.pooled, self.w, which=which)
        re = PanelRE(self.y, self.x, self.w, spat_diag=False)
        res = BSK_tests(re, self.w, which=which)
        np.testing.assert_allclose(res["Statistic"], exp["Statistic"], RTOL)

    def test_single_tests(self):
        full = BSK_tests(self.pooled, self.w, which=["LM1", "LM2", "LMJ"])
        for i, test in enumerate(["LM1", "LM2", "LMJ"]):
            res = BSK_tests(self.pooled, self.w, which=test)
            self.assertEqual(len(res), 1)
            np.testing.assert_allclose(
                res["Statistic"][0], full["Statistic"][i], RTOL
            )

    def test_no_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            PanelFE(self.y, self.x, self.w)
            PanelRE(self.y, self.x, self.w)
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
