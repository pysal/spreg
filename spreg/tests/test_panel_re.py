import unittest
import libpysal
import numpy as np
import pandas as pd
from spreg.panel_re import Panel_RE_Lag, Panel_RE_Error
from libpysal.common import RTOL
from libpysal.weights import w_subset


class Test_Panel_RE_Lag(unittest.TestCase):
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

    def test_Panel(self):
        reg = Panel_RE_Lag(
            self.y,
            self.x,
            w=self.w,
            name_y=self.y_name,
            name_x=self.x_names,
            name_ds=self.ds_name,
        )
        betas = np.array(
            [[4.44421994], [2.52821717], [2.24768846], [0.25846846], [0.68426639]]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([1.17169293])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([2.43910394])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        vm = np.array([0.08734092, 0.05232857, 0.05814063, 0.00164801, 0.00086908])
        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL)
        sig2 = np.array([[15.71234238]])
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.2634518198611293
        np.testing.assert_allclose(reg.pr2, pr2)
        std_err = np.array([0.29553498, 0.22875438, 0.24112368, 0.04059565, 0.02948021])
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        logll = -3127.652757262218
        np.testing.assert_allclose(reg.logll, logll, RTOL)
        aic = 6263.305514524436
        np.testing.assert_allclose(reg.aic, aic, RTOL)
        schwarz = 6283.3755390962015
        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)


class Test_Panel_RE_Error(unittest.TestCase):
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

#    def test_Panel(self):
#        reg = Panel_RE_Error(
#            self.y,
#            self.x,
#            w=self.w,
#            name_y=self.y_name,
#            name_x=self.x_names,
#            name_ds=self.ds_name,
#        )
#        betas = np.array(
#            [[5.87893756], [3.23269025], [2.62996804], [0.34042682], [4.9782446]]
#        )
#        np.testing.assert_allclose(reg.betas, betas, RTOL)
#        u = np.array([-0.2372652])
#        np.testing.assert_allclose(reg.u[0], u, RTOL)
#        predy = np.array([4.27277771])
#        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
#        vm = np.array([0.05163595, 0.05453637, 0.06134783, 0.00025012, 0.0030366])
#        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL)
#        sig2 = np.array([[16.10231419]])
#        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
#        pr2 = 0.3256008995950422
#        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
#        std_err = np.array([0.22723545, 0.23353024, 0.24768493, 0.01581518, 0.05510535])
#        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
#        logll = -7183.836220934392
#        np.testing.assert_allclose(reg.logll, logll, RTOL)
#        aic = 14373.672441868785
#        np.testing.assert_allclose(reg.aic, aic, RTOL)
#        schwarz = 14388.724960297608
#        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)


if __name__ == "__main__":
    unittest.main()
