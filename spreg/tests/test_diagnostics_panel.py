import unittest
import libpysal
import numpy as np
import pandas as pd
from spreg.diagnostics_panel import panel_LMlag, panel_LMerror, panel_rLMlag
from spreg.diagnostics_panel import panel_rLMerror, panel_Hausman
from spreg.panel_fe import Panel_FE_Lag, Panel_FE_Error
from spreg.panel_re import Panel_RE_Lag, Panel_RE_Error
from libpysal.common import RTOL
from libpysal.weights import w_subset


class Test_Panel_Diagnostics(unittest.TestCase):
    def setUp(self):
        self.ds_name = "NCOVR"
        nat = libpysal.examples.load_example(self.ds_name)
        self.db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
        nat_shp = libpysal.examples.get_path("NAT.shp")
        w_full = libpysal.weights.Queen.from_shapefile(nat_shp)
        self.y_name = ["HR70", "HR80", "HR90"]
        self.x_names = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
        c_names = ["STATE_NAME", "FIPSNO"]
        y_full = [self.db.by_col(name) for name in self.y_name]
        y_full = np.array(y_full).T
        x_full = [self.db.by_col(name) for name in self.x_names]
        x_full = np.array(x_full).T
        c_full = [self.db.by_col(name) for name in c_names]
        c_full = pd.DataFrame(c_full, index=c_names).T
        filter_states = ["Kansas", "Missouri", "Oklahoma", "Arkansas"]
        filter_counties = c_full[c_full["STATE_NAME"].isin(filter_states)]
        filter_counties = filter_counties["FIPSNO"].values
        counties = np.array(self.db.by_col("FIPSNO"))
        subid = np.where(np.isin(counties, filter_counties))[0]
        self.w = w_subset(w_full, subid)
        self.w.transform = "r"
        self.y = y_full[
            subid,
        ]
        self.x = x_full[
            subid,
        ]

    def test_LM(self):
        lmlag = panel_LMlag(self.y, self.x, self.w)
        exp = np.array([1.472807526666869, 0.22490325114767176])
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
        exp = np.array([-67.26822586935438, 1.0])
        np.testing.assert_allclose(Hlag, exp, RTOL)
#        Herror = panel_Hausman(fe_error, re_error)
#        exp = np.array([-84.38351088621853, 1.0])
#        np.testing.assert_allclose(Herror, exp, RTOL)


if __name__ == "__main__":
    unittest.main()
