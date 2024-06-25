import unittest
import libpysal
import numpy as np
from scipy import sparse
from spreg.ml_error import ML_Error
from libpysal.common import RTOL, ATOL
from warnings import filterwarnings

filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)
filterwarnings("ignore", message="^Method 'bounded' does not support")


class TestMLError(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("baltim.dbf"), "r")
        self.ds_name = "baltim.dbf"
        self.y_name = "PRICE"
        self.y = np.array(db.by_col(self.y_name)).T
        self.y.shape = (len(self.y), 1)
        self.x_names = ["NROOM", "AGE", "SQFT"]
        self.x = np.array([db.by_col(var) for var in self.x_names]).T
        ww = libpysal.io.open(libpysal.examples.get_path("baltim_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w_name = "baltim_q.gal"
        self.w.transform = "r"

    def _estimate_and_compare(self, method="FULL", RTOL=RTOL):
        reg = ML_Error(
            self.y,
            self.x,
            w=self.w,
            name_y=self.y_name,
            name_x=self.x_names,
            name_w=self.w_name,
            method=method,
        )
        betas = np.array([[19.45930348],
       [ 3.98928064],
       [-0.16714232],
       [ 0.57336871],
       [ 0.71757002]])
        np.testing.assert_allclose(reg.betas, betas, RTOL + 0.0001)
        u = np.array([29.870239])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([17.129761])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 211
        np.testing.assert_allclose(reg.n, n, RTOL)
        k = 4
        np.testing.assert_allclose(reg.k, k, RTOL)
        y = np.array([47.])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([  1.  ,   4.  , 148.  ,  11.25])
        np.testing.assert_allclose(reg.x[0], x, RTOL)
        e = np.array([44.392043])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        my = 44.30718
        np.testing.assert_allclose(reg.mean_y, my)
        sy = 23.606077
        np.testing.assert_allclose(reg.std_y, sy)
        vm = np.array(
            [3.775969e+01, 1.337534e+00, 4.440495e-03, 2.890193e-02,
            3.496050e-03]
        )
        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL)
        sig2 = np.array([[219.239799]])
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.341471
        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
        std_err = np.array(
            [6.144892, 1.156518, 0.066637, 0.170006, 0.059127]
        )
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        z_stat = [(3.166744811610107, 0.0015415552994677963),
                (3.4493895324306485, 0.0005618555635414317),
                (-2.5082495410045618, 0.012133094835810014),
                (3.3726442232925864, 0.0007445008419860677),
                (12.13599679437352, 6.807593113579489e-34)]
        np.testing.assert_allclose(reg.z_stat, z_stat, RTOL, atol=ATOL)
        logll = -881.269405
        np.testing.assert_allclose(reg.logll, logll, RTOL)
        aic = 1770.538809
        np.testing.assert_allclose(reg.aic, aic, RTOL)
        schwarz = 1783.946242
        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)
    def test_dense(self):
        self._estimate_and_compare(method="FULL")

    def test_LU(self):
        self._estimate_and_compare(method="LU", RTOL=RTOL * 10)

    def test_ord(self):
        reg = ML_Error(
            self.y,
            self.x,
            w=self.w,
            name_y=self.y_name,
            name_x=self.x_names,
            name_w=self.w_name,
            method="ORD",
        )
        betas = np.array([[19.45930348],
       [ 3.98928064],
       [-0.16714232],
       [ 0.57336871],
       [ 0.71757002]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([29.870239])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([17.129761])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 211
        np.testing.assert_allclose(reg.n, n, RTOL)
        k = 4
        np.testing.assert_allclose(reg.k, k, RTOL)
        y = np.array([47.])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([  1.  ,   4.  , 148.  ,  11.25])
        np.testing.assert_allclose(reg.x[0], x, RTOL)
        e = np.array([44.392043])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        my = 44.30718
        np.testing.assert_allclose(reg.mean_y, my)
        sy = 23.606077
        np.testing.assert_allclose(reg.std_y, sy)
        vm = np.array(
            [3.775969e+01, 1.337534e+00, 4.440495e-03, 2.890193e-02,
       3.586781e-03]
        )
        np.testing.assert_allclose(reg.vm.diagonal(), vm, RTOL * 10)
        sig2 = np.array([[219.239799]])
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.34147059826596426
        np.testing.assert_allclose(reg.pr2, pr2)
        std_err = np.array(
            [6.144892, 1.156518, 0.066637, 0.170006, 0.05989 ]
        )
        np.testing.assert_allclose(reg.std_err, std_err, RTOL * 10)
        z_stat = [(3.166744811610107, 0.0015415552994677963),
         (3.4493895324306485, 0.0005618555635414317),
         (-2.5082495410045618, 0.012133094835810014),
         (3.3726442232925864, 0.0007445008419860677),
         (11.981517603949666, 4.441183328428627e-33)]
        np.testing.assert_allclose(reg.z_stat, z_stat, rtol=RTOL, atol=ATOL)
        logll = -881.269405
        np.testing.assert_allclose(reg.logll, logll, RTOL)
        aic = 1770.538809
        np.testing.assert_allclose(reg.aic, aic, RTOL)
        schwarz = 1783.946242
        np.testing.assert_allclose(reg.schwarz, schwarz, RTOL)


if __name__ == "__main__":
    unittest.main()
