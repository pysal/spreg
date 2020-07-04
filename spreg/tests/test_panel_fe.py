import unittest
import libpysal
import numpy as np
from spreg.panel_fe import Panel_FE_Lag, Panel_FE_Error
from libpysal.common import RTOL


class Test_Panel_FE_Lag(unittest.TestCase):
    def setUp(self):
        self.ds_name = "NCOVR"
        nat = libpysal.examples.load_example(self.ds_name)
        self.db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
        nat_shp = libpysal.examples.get_path("NAT.shp")
        self.w = libpysal.weights.Queen.from_shapefile(nat_shp)
        self.w.transform = 'r'
        self.y_name = ["HR70", "HR80", "HR90"]
        self.x_names = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
        self.y = np.array([self.db.by_col(name) for name in self.y_name]).T
        self.x = np.array([self.db.by_col(name) for name in self.x_names]).T

    def test_Panel(self):
        reg = Panel_FE_Lag(self.y, self.x, w=self.w,
                           name_y=self.y_name, name_x=self.x_names,
                           name_ds=self.ds_name)
        betas = np.array([[0.80058859],
        [-2.60035236],
        [0.19030424]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-2.70317346])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-0.24876891])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        vm = np.array([0.02606527, 0.24359025, 0.00025597])
        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL)
        sig2 = np.array([[14.93535335]])
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.03191868031797557
        np.testing.assert_allclose(reg.pr2, pr2)
        std_err = np.array([0.16144743, 0.49354863, 0.01599908])
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        logll = -25673.94932137113
        np.testing.assert_allclose(reg.logll, logll, RTOL)
        aic = 51353.89864274226
        np.testing.assert_allclose(reg.aic, aic, RTOL)
        schwarz = 51375.29740041728
        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)


class Test_Panel_FE_Error(unittest.TestCase):
    def setUp(self):
        self.ds_name = "NCOVR"
        nat = libpysal.examples.load_example(self.ds_name)
        self.db = libpysal.io.open(nat.get_path("NAT.dbf"), "r")
        nat_shp = libpysal.examples.get_path("NAT.shp")
        self.w = libpysal.weights.Queen.from_shapefile(nat_shp)
        self.w.transform = 'r'
        self.y_name = ["HR70", "HR80", "HR90"]
        self.x_names = ["RD70", "RD80", "RD90", "PS70", "PS80", "PS90"]
        self.y = np.array([self.db.by_col(name) for name in self.y_name]).T
        self.x = np.array([self.db.by_col(name) for name in self.x_names]).T

    def test_Panel(self):
        reg = Panel_FE_Error(self.y, self.x, w=self.w,
                             name_y=self.y_name, name_x=self.x_names,
                             name_ds=self.ds_name)
        betas = np.array([[0.86979232],
                          [-2.96606744],
                          [0.19434604]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-0.07023431])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-2.88170807])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        vm = np.array([1.36380130e-01, 1.36978101e+00, 2.56811350e-04])
        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL)
        sig2 = np.array([[68.95139869]])
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 3.662469318784559e-05
        np.testing.assert_allclose(reg.pr2, pr2)
        std_err = np.array([0.3692968 , 1.17037644, 0.01602533])
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        logll = -35839.07814595635
        np.testing.assert_allclose(reg.logll, logll, RTOL)
        aic = 71682.1562919127
        np.testing.assert_allclose(reg.aic, aic, RTOL)
        schwarz = 71696.42213036271
        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)


if __name__ == '__main__':
    unittest.main()
