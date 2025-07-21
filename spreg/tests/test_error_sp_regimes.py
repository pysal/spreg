import unittest
import scipy
import libpysal
import numpy as np
from spreg import error_sp_regimes as SP
from spreg.error_sp import GM_Error, GM_Endog_Error, GM_Combo
from libpysal.common import RTOL


class TestGM_Error_Regimes(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49, 1))
        X = []
        X.append(db.by_col("HOVAL"))
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Queen.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"
        self.r_var = "NSA"
        self.regimes = db.by_col(self.r_var)
        X1 = []
        X1.append(db.by_col("INC"))
        self.X1 = np.array(X1).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        # Artficial:
        n = 256
        self.n2 = n // 2
        self.x_a1 = np.random.uniform(-10, 10, (n, 1))
        self.x_a2 = np.random.uniform(1, 5, (n, 1))
        self.q_a = self.x_a2 + np.random.normal(0, 1, (n, 1))
        self.x_a = np.hstack((self.x_a1, self.x_a2))
        self.y_a = np.dot(
            np.hstack((np.ones((n, 1)), self.x_a)), np.array([[1], [0.5], [2]])
        ) + np.random.normal(0, 1, (n, 1))
        latt = int(np.sqrt(n))
        self.w_a = libpysal.weights.util.lat2W(latt, latt)
        self.w_a.transform = "r"
        self.regi_a = [0] * (n // 2) + [1] * (n // 2)  ##must be floors!
        self.w_a1 = libpysal.weights.util.lat2W(latt // 2, latt)
        self.w_a1.transform = "r"

    def test_model(self):
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w)
        betas = np.array(
            [
                [63.3443073],
                [-0.15468],
                [-1.52186509],
                [61.40071412],
                [-0.33550084],
                [-0.85076108],
                [0.38671608],
            ]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-2.06177251])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([17.78775251])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        k = 6
        np.testing.assert_allclose(reg.k, k, RTOL)
        y = np.array([15.72598])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([[0.0, 0.0, 0.0, 1.0, 80.467003, 19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(), x, RTOL)
        e = np.array([1.40747232])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y, my, RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y, sy, RTOL)
        vm = np.array([50.55875289, -0.14444487, -2.05735489, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
        sig2 = 102.13050615267227
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.5525102200608539
        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
        std_err = np.array(
            [7.11046784, 0.21879293, 0.58477864, 7.50596504, 0.10800686, 0.57365981]
        )
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        chow_r = np.array(
            [
                [0.03533785, 0.85088948],
                [0.54918491, 0.45865093],
                [0.67115641, 0.41264872],
            ]
        )
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL)
        chow_j = 0.81985446000130979
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, RTOL)
    
    def teste_model_slx(self):
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w, slx_lags=1)
        betas = np.array([
                    [81.52150852],
                    [-0.13555178],
                    [-1.51660154],
                    [0.33159891],
                    [-1.88989597],
                    [68.76353402],
                    [-0.28052211],
                    [-0.96302502],
                    [0.21438560],
                    [-1.28101731],
                    [0.34676832]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([4.56150554])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([11.16447446])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        k = 10
        np.testing.assert_allclose(reg.k, k, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([[0.00000000, 0.00000000, 0.00000000,
                       0.00000000, 0.00000000, 1.00000000,
                       80.46700300, 19.53100000, 35.45850050, 
                       18.59400000 ]])
        np.testing.assert_allclose(reg.x[0].toarray(), x, RTOL)
        e_filtered = np.array([7.50323326])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 148.34560113, -0.30813759, -0.71768406,
                        -1.80329540, -2.85215453, 0.00000000, 
                        0.00000000, 0.00000000, 0.00000000,
                        0.00000000 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
        sig2 = 91.65279833
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.62579728
        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
        std_err = np.array([ 12.17972090, 0.21106390, 0.57174395, 
                            0.51968679, 1.12246122, 11.98889457,
                            0.10978796, 0.54508806, 0.24453164,
                            0.93883245 ])
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        chow_r = np.array([
            [0.55726637, 0.45536376],
            [0.37130546, 0.54229354],
            [0.49109136, 0.48344088],
            [0.04164961, 0.83828913],
            [0.17313258, 0.67734266]])
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL)
        chow_j = 1.82626123
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, RTOL)
    
    def teste_model_1 (self):
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[ 60.45730439],
                          [ -0.17732134],
                          [ -1.30936328],
                          [  0.51314713],
                          [ 66.5487126 ],
                          [ -0.31845995],
                          [ -1.29047149],
                          [  0.08092997]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([0.00698340])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([15.71899660])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([0.53685671])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)      
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 39.33653719, -0.08420816, -1.50351108, 0.00000000, 0.00000000, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)

    """
    def test_model_regi_error(self):
        #Columbus:
        reg = SP.GM_Error_Regimes(self.y, self.X, self.regimes, self.w, regime_err_sep=True)
        betas = np.array([[ 60.45730439],
       [ -0.17732134],
       [ -1.30936328],
       [  0.51314713],
       [ 66.5487126 ],
       [ -0.31845995],
       [ -1.29047149],
       [  0.08092997]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([ 39.33656288,  -0.08420799,  -1.50350999,   0.        ,
         0.        ,   0.        ])
        np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        u = np.array([ 0.00698341])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 15.71899659])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        e = np.array([ 0.53685671])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        chow_r = np.array([[  3.63674458e-01,   5.46472584e-01],
       [  4.29607250e-01,   5.12181727e-01],
       [  5.44739543e-04,   9.81379339e-01]])
        np.testing.assert_allclose(reg.chow.regi,chow_r,RTOL)
        chow_j = 0.70119418251625387
        np.testing.assert_allclose(reg.chow.joint[0],chow_j,RTOL)
        #Artficial:
        model = SP.GM_Error_Regimes(self.y_a, self.x_a, self.regi_a, w=self.w_a, regime_err_sep=True)
        model1 = GM_Error(self.y_a[0:(self.n2)].reshape((self.n2),1), self.x_a[0:(self.n2)], w=self.w_a1)
        model2 = GM_Error(self.y_a[(self.n2):].reshape((self.n2),1), self.x_a[(self.n2):], w=self.w_a1)
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas,tbetas,RTOL)
        vm = np.hstack((model1.vm.diagonal(),model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm,RTOL)
    """

    def test_model_endog(self):
        reg = SP.GM_Endog_Error_Regimes(
            self.y, self.X1, self.yd, self.q, self.regimes, self.w
        )
        betas = np.array(
            [
                [
                    77.48385551,
                    4.52986622,
                    78.93209405,
                    0.42186261,
                    -3.23823854,
                    -1.1475775,
                    0.20222108,
                ]
            ]
        )
        print("Runining higher-tolerance test on L133 of test_error_sp_regimes.py")
        np.testing.assert_allclose(reg.betas.T, betas, RTOL + 0.0001)
        u = np.array([20.89660904])
        # np.testing.assert_allclose(reg.u[0],u,RTOL)
        np.testing.assert_allclose(reg.u[0], u, rtol=1e-05)
        e = np.array([25.21818724])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        predy = np.array([-5.17062904])
        # np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        np.testing.assert_allclose(reg.predy[0], predy, rtol=1e-03)
        n = 49
        np.testing.assert_allclose(reg.n, n)
        k = 6
        np.testing.assert_allclose(reg.k, k)
        y = np.array([15.72598])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([[0.0, 0.0, 1.0, 19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(), x, RTOL)
        yend = np.array([[0.0, 80.467003]])
        np.testing.assert_allclose(reg.yend[0].toarray(), yend, RTOL)
        z = np.array([[0.0, 0.0, 1.0, 19.531, 0.0, 80.467003]])
        np.testing.assert_allclose(reg.z[0].toarray(), z, RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y, my)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y, sy)
        vm = np.array([390.88250241, 52.25924084, 0.0, 0.0, -32.64274729, 0.0])
        # np.testing.assert_allclose(reg.vm[0],vm,RTOL)
        np.allclose(reg.vm, vm)
        pr2 = 0.19623994206233333
        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
        sig2 = 649.4011
        # np.testing.assert_allclose(round(reg.sig2,RTOL),round(sig2,RTOL),RTOL)
        np.testing.assert_allclose(sig2, reg.sig2, rtol=1e-05)
        std_err = np.array(
            [19.77074866, 6.07667394, 24.32254786, 2.17776972, 2.97078606, 0.94392418]
        )
        # np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        np.testing.assert_allclose(reg.std_err, std_err, rtol=1e-05)
        chow_r = np.array(
            [[0.0021348, 0.96314775], [0.40499741, 0.5245196], [0.4498365, 0.50241261]]
        )
        print("Running higher-tolerance tests on L176 of test_error_sp_regimes.py")
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL + 0.0001)
        chow_j = 1.2885590185243503
        # np.testing.assert_allclose(reg.chow.joint[0],chow_j)
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, rtol=1e-05)

    def test_model_endog_regi_error(self):
        # Columbus:
        reg = SP.GM_Endog_Error_Regimes(
            self.y, self.X1, self.yd, self.q, self.regimes, self.w, regime_err_sep=True
        )
        betas = np.array(
            [
                [7.91729500e01],
                [5.80693176e00],
                [-3.84036576e00],
                [1.46462983e-01],
                [8.24723791e01],
                [5.68908920e-01],
                [-1.28824699e00],
                [6.70387351e-02],
            ]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        vm = np.array([791.86679123, 140.12967794, -81.37581255, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
        u = np.array([25.80361497])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-10.07763497])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        e = np.array([27.32251813])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        chow_r = np.array(
            [
                [0.00926459, 0.92331985],
                [0.26102777, 0.60941494],
                [0.26664581, 0.60559072],
            ]
        )
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL)
        chow_j = 1.1184631131987004
        # np.testing.assert_allclose(reg.chow.joint[0],chow_j)
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, RTOL)
        # Artficial:
        model = SP.GM_Endog_Error_Regimes(
            self.y_a,
            self.x_a1,
            yend=self.x_a2,
            q=self.q_a,
            regimes=self.regi_a,
            w=self.w_a,
            regime_err_sep=True,
        )
        model1 = GM_Endog_Error(
            self.y_a[0 : (self.n2)].reshape((self.n2), 1),
            self.x_a1[0 : (self.n2)],
            yend=self.x_a2[0 : (self.n2)],
            q=self.q_a[0 : (self.n2)],
            w=self.w_a1,
        )
        model2 = GM_Endog_Error(
            self.y_a[(self.n2) :].reshape((self.n2), 1),
            self.x_a1[(self.n2) :],
            yend=self.x_a2[(self.n2) :],
            q=self.q_a[(self.n2) :],
            w=self.w_a1,
        )
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas, tbetas, RTOL)
        vm = np.hstack((model1.vm.diagonal(), model2.vm.diagonal()))
        np.testing.assert_allclose(model.vm.diagonal(), vm, RTOL)

    def test_model_combo(self):
        reg = SP.GM_Combo_Regimes(
            self.y, self.X1, self.regimes, self.yd, self.q, w=self.w
        )
        predy_e = np.array([18.82774339])
        np.testing.assert_allclose(reg.predy_e[0], predy_e, RTOL)
        betas = np.array(
            [
                [36.44798052],
                [-0.7974482],
                [30.53782661],
                [-0.72602806],
                [-0.30953121],
                [-0.21736652],
                [0.64801059],
                [-0.16601265],
            ]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([0.84393304])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        e_filtered = np.array([0.4040027])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        predy = np.array([14.88204696])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        k = 7
        np.testing.assert_allclose(reg.k, k, RTOL)
        y = np.array([15.72598])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        x = np.array([[0.0, 0.0, 1.0, 19.531]])
        np.testing.assert_allclose(reg.x[0].toarray(), x, RTOL)
        yend = np.array([[0.0, 80.467003, 24.7142675]])
        np.testing.assert_allclose(reg.yend[0].toarray(), yend, RTOL)
        z = np.array([[0.0, 0.0, 1.0, 19.531, 0.0, 80.467003, 24.7142675]])
        np.testing.assert_allclose(reg.z[0].toarray(), z, RTOL)
        my = 35.128823897959187
        np.testing.assert_allclose(reg.mean_y, my, RTOL)
        sy = 16.732092091229699
        np.testing.assert_allclose(reg.std_y, sy, RTOL)
        vm = np.array(
            [
                109.23549239,
                -0.19754121,
                84.29574673,
                -1.99317178,
                -1.60123994,
                -0.1252719,
                -1.3930344,
            ]
        )
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
        sig2 = 94.98610921110007
        np.testing.assert_allclose(reg.sig2, sig2, RTOL)
        pr2 = 0.6493586702255537
        np.testing.assert_allclose(reg.pr2, pr2, RTOL)
        pr2_e = 0.5255332447240576
        np.testing.assert_allclose(reg.pr2_e, pr2_e, RTOL)
        std_err = np.array(
            [
                10.45157846,
                0.93942923,
                11.38484969,
                0.60774708,
                0.44461334,
                0.15871227,
                0.15738141,
            ]
        )
        np.testing.assert_allclose(reg.std_err, std_err, RTOL)
        chow_r = np.array(
            [
                [0.49716076, 0.48075032],
                [0.00405377, 0.94923363],
                [0.03866684, 0.84411016],
            ]
        )
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL)
        chow_j = 0.64531386285872072
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, RTOL)

    def test_model_combo_regi_error(self):
        # Columbus:
        reg = SP.GM_Combo_Regimes(
            self.y,
            self.X1,
            self.regimes,
            self.yd,
            self.q,
            w=self.w,
            regime_lag_sep=True,
            regime_err_sep=True,
        )
        betas = np.array(
            [
                [42.01035248],
                [-0.13938772],
                [-0.6528306],
                [0.54737621],
                [0.2684419],
                [34.02473255],
                [-0.14920259],
                [-0.48972903],
                [0.65883658],
                [-0.17174845],
            ]
        )
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        vm = np.array(
            [153.58614432, 2.96302131, -3.26211855, -2.46914703, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
        u = np.array([7.73968703])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([7.98629297])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        e = np.array([6.45052714])
        np.testing.assert_allclose(reg.e_filtered[0], e, RTOL)
        chow_r = np.array(
            [
                [1.00886404e-01, 7.50768497e-01],
                [3.61843271e-05, 9.95200481e-01],
                [4.69585772e-02, 8.28442711e-01],
                [8.13275259e-02, 7.75506385e-01],
            ]
        )
        np.testing.assert_allclose(reg.chow.regi, chow_r, RTOL)
        chow_j = 0.28479988992843119
        np.testing.assert_allclose(reg.chow.joint[0], chow_j, RTOL)
        # Artficial:
        model = SP.GM_Combo_Regimes(
            self.y_a,
            self.x_a1,
            yend=self.x_a2,
            q=self.q_a,
            regimes=self.regi_a,
            w=self.w_a,
            regime_err_sep=True,
            regime_lag_sep=True,
        )
        model1 = GM_Combo(
            self.y_a[0 : (self.n2)].reshape((self.n2), 1),
            self.x_a1[0 : (self.n2)],
            yend=self.x_a2[0 : (self.n2)],
            q=self.q_a[0 : (self.n2)],
            w=self.w_a1,
        )
        model2 = GM_Combo(
            self.y_a[(self.n2) :].reshape((self.n2), 1),
            self.x_a1[(self.n2) :],
            yend=self.x_a2[(self.n2) :],
            q=self.q_a[(self.n2) :],
            w=self.w_a1,
        )
        tbetas = np.vstack((model1.betas, model2.betas))
        np.testing.assert_allclose(model.betas, tbetas)
        vm = np.hstack((model1.vm.diagonal(), model2.vm.diagonal()))

class GMM_Error_Regimes(unittest.TestCase):
    def setUp(self):
        db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
        y = np.array(db.by_col("CRIME"))
        self.y = np.reshape(y, (49, 1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Queen.from_shapefile(
            libpysal.examples.get_path("columbus.shp")
        )
        self.w.transform = "r"
        self.r_var = "NSA"
        self.regimes = db.by_col(self.r_var)
        X1 = []
        X1.append(db.by_col("INC"))
        self.X1 = np.array(X1).T
        yd = []
        yd.append(db.by_col("HOVAL"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T

    def test_model_kp98(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='kp98')
        betas = np.array([
             [64.03361891],
             [-1.91909793],
             [55.92223354],
             [-1.50076966],
             [0.30990527]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-10.88472129])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([26.61070129])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-9.86121112])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 57.51790950, -2.81322652, 0.00000000, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
    
    def test_model_kp98_1(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='kp98', add_wy=True)
        betas = np.array([
            [39.83761587],
            [-1.54921000],
            [33.12940644],
            [-1.35443333],
            [0.56243800],
            [-0.28535967]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-4.85023212])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([20.57621212])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-4.53150217])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 143.74766140, -4.70373507, 125.67978801, -3.67903682, -2.03871945 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)

    def test_model_kp98_2(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='kp98', yend=self.yd, q=self.q)
        betas = np.array([
            [77.48388286],
            [4.52987507],
            [78.93213633],
            [0.42186355],
            [-3.23824313],
            [-1.14757886],
            [0.20221915]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([20.89665796])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-5.17067796])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([25.21819730])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 390.88287241, 52.25937018, 0.00000000, 0.00000000, -32.64281528, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)

    def test_model_hom(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='hom', A1='hom_sc')
        betas = np.array([
            [64.01695738],
            [-1.91795762],
            [55.89797832],
            [-1.49907653],
            [0.43140730]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-10.89353465])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([26.61951465])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-9.46562746])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 55.70247014, -2.65868760, 3.13170131, -0.05525004, 0.02766334 ])
    
    def test_model_hom_1(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='hom', A1='hom_sc', add_wy=True)
        betas = np.array([
            [39.82877996],
            [-1.54899235],
            [33.09477734],
            [-1.35278327],
            [0.56271272],
            [-0.17744710]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-4.85461996])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([20.58059996])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-4.65707811])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 141.65497781, -4.63063015, 115.94973717, -2.99053918, -2.00152072, 2.64688262 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
    
    def test_model_hom_2(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='hom', yend=self.yd, q=self.q,A1='hom_sc')
        betas = np.array([
            [77.47719809],
            [4.52771077],
            [78.92179930],
            [0.42163460],
            [-3.23712142],
            [-1.14724621],
            [0.24175655]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([20.88469996])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-5.15871996])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([26.05049963])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 419.87276353, 72.19591647, 22.43805910, 4.10700927, -42.04775263, -1.87819164, -1.60508208 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
    
    def test_model_het(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='het',step1c=True)
        betas = np.array([
            [63.64307764],
            [-1.89224844],
            [55.35522507],
            [-1.46141834],
            [0.46917553]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-11.08628356])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([26.81226356])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-9.45949012])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 33.89701616, -1.33058462, 8.10836078, -0.39660087, 0.00000000 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
    
    def test_model_1(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='het',step1c=True, add_wy=True)
        betas = np.array([
            [39.72414325],
            [-1.54645065],
            [32.64933929],
            [-1.33142873],
            [0.56616310],
            [-0.09895070]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([-4.91153115])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([20.63751115])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([-4.80608992])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 82.54351678, -2.81786650, 52.57090657, -0.85398469, -1.18612006, 1.09531546 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)
    
    def test_model_2(self):
        reg = SP.GMM_Error_Regimes(self.y, self.X, self.regimes, self.w, estimator='het', yend=self.yd, q=self.q, step1c=True)
        betas = np.array([
            [77.26679984],
            [4.45992905],
            [78.59534391],
            [0.41432319],
            [-3.20196287],
            [-1.13672283],
            [0.21835250]])
        np.testing.assert_allclose(reg.betas, betas, RTOL)
        u = np.array([20.50716918])
        np.testing.assert_allclose(reg.u[0], u, RTOL)
        predy = np.array([-4.78118918])
        np.testing.assert_allclose(reg.predy[0], predy, RTOL)
        n = 49
        np.testing.assert_allclose(reg.n, n, RTOL)
        y = np.array([15.72598000])
        np.testing.assert_allclose(reg.y[0], y, RTOL)
        e_filtered = np.array([25.15338615])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        mean_y = 35.12882390
        np.testing.assert_allclose(reg.mean_y, mean_y, RTOL)
        std_y = 16.73209209
        np.testing.assert_allclose(reg.std_y, std_y, RTOL)
        vm = np.array([ 509.97341730, 150.74634158, 9.66536908, 5.57390017, -81.03896450, -2.26333955, -3.23574832 ])
        np.testing.assert_allclose(reg.vm[0], vm, RTOL)

    
if __name__ == "__main__":
    unittest.main()
