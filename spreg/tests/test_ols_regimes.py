import unittest
import numpy as np
import libpysal
from spreg.ols import OLS
from spreg.ols_regimes import OLS_Regimes
from libpysal.common import RTOL

PEGP = libpysal.examples.get_path


class TestOLS_regimes(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        self.y_var = "CRIME"
        self.y = np.array([db.by_col(self.y_var)]).reshape(49, 1)
        self.x_var = ["INC", "HOVAL"]
        self.x = np.array([db.by_col(name) for name in self.x_var]).T
        self.r_var = "NSA"
        self.regimes = db.by_col(self.r_var)
        self.w = libpysal.weights.Rook.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"

    def test_OLS(self):
        start_suppress = np.get_printoptions()["suppress"]
        np.set_printoptions(suppress=True)
        ols = OLS_Regimes(
            self.y,
            self.x,
            self.regimes,
            w=self.w,
            regime_err_sep=False,
            constant_regi="many",
            nonspat_diag=False,
            spat_diag=True,
            name_y=self.y_var,
            name_x=self.x_var,
            name_ds="columbus",
            name_regimes=self.r_var,
            name_w="columbus.gal",
        )
        # np.testing.assert_allclose(ols.aic, 408.73548964604873 ,RTOL)
        np.testing.assert_allclose(ols.ar2, 0.50761700679873101, RTOL)
        np.testing.assert_allclose(
            ols.betas,
            np.array(
                [
                    [68.78670869],
                    [-1.9864167],
                    [-0.10887962],
                    [67.73579559],
                    [-1.36937552],
                    [-0.31792362],
                ]
            ),
        )
        vm = np.array([48.81339213, -2.14959579, -0.19968157, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(ols.vm[0], vm, RTOL)
        np.testing.assert_allclose(ols.lm_error, (5.92970357, 0.01488775), RTOL)
        np.testing.assert_allclose(ols.lm_lag, (8.78315751, 0.00304024), RTOL)
        np.testing.assert_allclose(ols.lm_sarma, (8.89955982, 0.01168114), RTOL)
        np.testing.assert_allclose(ols.mean_y, 35.1288238979591, RTOL)
        np.testing.assert_equal(ols.k, 6)
        np.testing.assert_equal(ols.kf, 0)
        np.testing.assert_equal(ols.kr, 3)
        np.testing.assert_equal(ols.n, 49)
        np.testing.assert_equal(ols.nr, 2)
        np.testing.assert_equal(ols.name_ds, "columbus")
        np.testing.assert_equal(ols.name_gwk, None)
        np.testing.assert_equal(ols.name_w, "columbus.gal")
        np.testing.assert_equal(
            ols.name_x,
            ["0_CONSTANT", "0_INC", "0_HOVAL", "1_CONSTANT", "1_INC", "1_HOVAL"],
        )
        np.testing.assert_equal(ols.name_y, "CRIME")
        np.testing.assert_allclose(ols.predy[3], np.array([51.05003696]), RTOL)
        np.testing.assert_allclose(ols.r2, 0.55890690192386316, RTOL)
        np.testing.assert_allclose(ols.rlm_error, (0.11640231, 0.73296972), RTOL)
        np.testing.assert_allclose(ols.rlm_lag, (2.96985625, 0.08482939), RTOL)
        np.testing.assert_equal(ols.robust, "unadjusted")
        np.testing.assert_allclose(ols.sig2, 137.84897351821013, RTOL)
        np.testing.assert_allclose(ols.sig2n, 120.96950737312316, RTOL)
        np.testing.assert_allclose(ols.t_stat[2][0], -0.43342216706091791, RTOL)
        np.testing.assert_allclose(ols.t_stat[2][1], 0.66687472578594531, RTOL)
        np.set_printoptions(suppress=start_suppress)

    """
    def test_OLS_regi(self):
        #Artficial:
        n = 256
        x1 = np.random.uniform(-10,10,(n,1))
        y = np.dot(np.hstack((np.ones((n,1)),x1)),np.array([[1],[0.5]])) + np.random.normal(0,1,(n,1))
        latt = int(np.sqrt(n))
        regi = [0]*(n/2) + [1]*(n/2)
        model = OLS_Regimes(y, x1, regimes=regi, regime_err_sep=True, sig2n_k=False)
        model1 = OLS(y[0:(n/2)].reshape((n/2),1), x1[0:(n/2)], sig2n_k=False)
        model2 = OLS(y[(n/2):n].reshape((n/2),1), x1[(n/2):n], sig2n_k=False)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm,RTOL)
        #Columbus:  
        reg = OLS_Regimes(self.y, self.x, self.regimes, w=self.w, constant_regi='many', nonspat_diag=True, spat_diag=True, name_y=self.y_var, name_x=self.x_var, name_ds='columbus', name_regimes=self.r_var, name_w='columbus.gal', regime_err_sep=True)        
        np.testing.assert_allclose(reg.multi[0].aic, 192.96044303402897 ,RTOL)
        tbetas = np.array([[ 68.78670869],
       [ -1.9864167 ],
       [ -0.10887962],
       [ 67.73579559],
       [ -1.36937552],
       [ -0.31792362]])
        np.testing.assert_allclose(tbetas, reg.betas,RTOL)
        vm = np.array([ 41.68828023,  -1.83582717,  -0.17053478,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0], vm,RTOL)
        u_3 = np.array([[ 0.31781838],
       [-5.6905584 ],
       [-6.8819715 ]])
        np.testing.assert_allclose(reg.u[0:3], u_3,RTOL)
        predy_3 = np.array([[ 15.40816162],
       [ 24.4923124 ],
       [ 37.5087525 ]])
        np.testing.assert_allclose(reg.predy[0:3], predy_3,RTOL)
        chow_regi = np.array([[ 0.01002733,  0.92023592],
       [ 0.46017009,  0.49754449],
       [ 0.60732697,  0.43579603]])
        np.testing.assert_allclose(reg.chow.regi, chow_regi,RTOL)
        np.testing.assert_allclose(reg.chow.joint[0], 0.67787986791767096,RTOL)
    """

    def test_OLS_fixed(self):
        start_suppress = np.get_printoptions()["suppress"]
        np.set_printoptions(suppress=True)
        ols = OLS_Regimes(
            self.y,
            self.x,
            self.regimes,
            w=self.w,
            cols2regi=[False, True],
            regime_err_sep=True,
            constant_regi="one",
            nonspat_diag=False,
            spat_diag=True,
            name_y=self.y_var,
            name_x=self.x_var,
            name_ds="columbus",
            name_regimes=self.r_var,
            name_w="columbus.gal",
        )
        np.testing.assert_allclose(
            ols.betas,
            np.array([[-0.24385565], [-0.26335026], [68.89701137], [-1.67389685]]),
            RTOL,
        )
        vm = np.array([0.02354271, 0.01246677, 0.00424658, -0.04921356])
        np.testing.assert_allclose(ols.vm[0], vm, RTOL)
        np.testing.assert_allclose(ols.lm_error, (5.62668744, 0.01768903), RTOL)
        np.testing.assert_allclose(ols.lm_lag, (9.43264957, 0.00213156), RTOL)
        np.testing.assert_allclose(ols.mean_y, 35.12882389795919, RTOL)
        np.testing.assert_equal(ols.kf, 2)
        np.testing.assert_equal(ols.kr, 1)
        np.testing.assert_equal(ols.n, 49)
        np.testing.assert_equal(ols.nr, 2)
        np.testing.assert_equal(ols.name_ds, "columbus")
        np.testing.assert_equal(ols.name_gwk, None)
        np.testing.assert_equal(ols.name_w, "columbus.gal")
        np.testing.assert_equal(
            ols.name_x, ["0_HOVAL", "1_HOVAL", "_Global_CONSTANT", "_Global_INC"]
        )
        np.testing.assert_equal(ols.name_y, "CRIME")
        np.testing.assert_allclose(ols.predy[3], np.array([52.65974636]), RTOL)
        np.testing.assert_allclose(ols.r2, 0.5525561183786056, RTOL)
        np.testing.assert_equal(ols.robust, "unadjusted")
        np.testing.assert_allclose(ols.t_stat[2][0], 13.848705206463748, RTOL)
        np.testing.assert_allclose(ols.t_stat[2][1], 7.776650625274256e-18, RTOL)
        np.set_printoptions(suppress=start_suppress)


if __name__ == "__main__":
    unittest.main()
