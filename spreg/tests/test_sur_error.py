import unittest
import numpy as np
import libpysal
import geopandas as gpd
from spreg.sur_utils import sur_dictxy
from spreg.sur_error import SURerrorML, SURerrorGM
from libpysal.common import RTOL

ATOL = 0.0001

def dict_compare(actual, desired, rtol, atol=1e-7):
    for i in actual.keys():
        np.testing.assert_allclose(actual[i], desired[i], rtol, atol=atol)

class Test_SUR_error(unittest.TestCase):
    def setUp(self):
        nat = libpysal.examples.load_example('NCOVR')
        self.db = gpd.read_file(nat.get_path("NAT.shp"))
        self.dbs = self.db[self.db['SOUTH'] == 1]
        self.w = libpysal.weights.Queen.from_dataframe(self.dbs)
        self.w.transform = 'r'

    def test_error(self):  # 2 equations
        y_var0 = ["HR80", "HR90"]
        x_var0 = [["PS80", "UE80"], ["PS90", "UE90"]]
        reg = SURerrorML(
            y_var0,
            x_var0,
            self.w,
            df=self.dbs,
            spat_diag=True,
            vm=True,
            name_w="natqueen",
            name_ds="nat",
            nonspat_diag=True,
        )

        dict_compare(
            reg.bSUR0,
            {
                0: np.array([[8.460653],
            [1.139978],
            [0.250482]]),
                1: np.array([[6.82362 ],
            [1.115422],
            [0.36281 ]]),
            },
            RTOL,
        )
        dict_compare(
            reg.bSUR,
            {
                0: np.array([[7.686057],
            [1.277312],
            [0.358611]]),
                1: np.array([[6.061804],
            [1.176992],
            [0.466222]]),
            },
            RTOL,
        )
        dict_compare(
            reg.sur_inf,
            {
                0: np.array(
                    [[5.231980e-01, 1.469053e+01, 7.412822e-49],
            [2.210486e-01, 5.778424e+00, 7.540380e-09],
            [6.830534e-02, 5.250110e+00, 1.520084e-07]]
                ),
                1: np.array(
                    [[5.615405e-01, 1.079495e+01, 3.636453e-27],
            [2.374752e-01, 4.956272e+00, 7.185859e-07],
            [6.682544e-02, 6.976722e+00, 3.021471e-12]]
                ),
            },
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lamols, np.array([[0.452342],
            [0.456641]]), RTOL
        )
        np.testing.assert_allclose(
            reg.lamsur, np.array([[0.406481],
            [0.404923]]), RTOL
        )
        np.testing.assert_allclose(
            reg.corr, np.array([[1.      , 0.317296],
            [0.317296, 1.      ]]), RTOL
        )
        np.testing.assert_allclose(
            reg.surchow,
        [[5.754872, 1.      , 0.016443],
            [0.139452, 1.      , 0.708826],
            [1.565087, 1.      , 0.210922]],
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(reg.llik, -9210.502749)
        np.testing.assert_allclose(reg.errllik, -9154.633582)
        np.testing.assert_allclose(reg.surerrllik, -9084.265703)
        np.testing.assert_allclose(
            reg.likrlambda, ([2.524741e+02, 2.000000e+00, 1.499504e-55])
        )
        np.testing.assert_allclose(
            reg.vm,
            np.array(
        [[ 1.15080293e-03,  6.37266403e-05, -7.11447519e-03,
                -1.26916425e-03, -4.45217262e-04],
            [ 6.37266403e-05,  1.15389186e-03, -3.93969800e-04,
                -1.26666791e-03, -8.06150412e-03],
            [-7.11447519e-03, -3.93969800e-04,  1.76804521e+00,
                5.90841784e-01,  1.99893702e-01],
            [-1.26916425e-03, -1.26666791e-03,  5.90841784e-01,
                1.08029750e+00,  6.71007645e-01],
            [-4.45217262e-04, -8.06150412e-03,  1.99893702e-01,
                6.71007645e-01,  2.28037800e+00]]
            ),
            RTOL,
        )
        np.testing.assert_allclose(
            reg.lamsetp,
        (np.array([[0.03392349],
                [0.03396898]]),
        np.array([[11.98229414],
                [11.92037038]]),
        np.array([[4.39977145e-33],
                [9.26955308e-33]])),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.joinlam, [2.707006e+02, 2.000000e+00, 1.652354e-59], rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            reg.likrlambda,
            [2.524741e+02, 2.000000e+00, 1.499504e-55],
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lrtest,
            [1.407358e+02, 1.000000e+00, 1.837906e-32],
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lamtest,
            [0.001115, 1.      , 0.973358],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_error_3eq(self):  # Three equation example, unequal K
        bigy1b, bigX1b, bigyvars1, bigXvars1 = sur_dictxy(self.dbs, ["HR60", "HR70", "HR80"], [["RD60", "PS60"], ["RD70", "PS70", "UE70"], ["RD80", "PS80"]])
        reg = SURerrorML(
            bigy1b,
            bigX1b,
            self.w
        )

        dict_compare(
            reg.bSUR0,
            {
                0: np.array([[6.176503],
        [1.57968 ],
        [0.487517]]),
                1: np.array([[10.393457],
        [ 2.847285],
        [ 1.175123],
        [-0.448833]]),
                2: np.array([[8.352587],
        [2.911434],
        [1.9949  ]]),
            },
            RTOL,
        )
        dict_compare(
            reg.bSUR,
            {
                0: np.array([[6.178485],
        [1.550605],
        [0.400335]]),
                1: np.array([[ 9.961855],
        [ 2.704641],
        [ 0.917635],
        [-0.328154]]),
                2: np.array([[8.245629],
        [3.045425],
        [1.975781]]),
            },
            RTOL,
        )
        dict_compare(
            reg.sur_inf,
            {
                0: np.array(
                    [[2.745844e-001, 2.250122e+001, 4.037644e-112],
            [2.196472e-001, 7.059523e+000, 1.670748e-012],
            [2.277493e-001, 1.757788e+000, 7.878356e-002]]
                ),
                1: np.array(
                    [[ 5.209554e-01,  1.912228e+01,  1.647484e-81],
            [ 2.664617e-01,  1.015021e+01,  3.306623e-24],
            [ 2.731269e-01,  3.359740e+00,  7.801591e-04],
            [ 1.073297e-01, -3.057441e+00,  2.232355e-03]]
                ),
                2: np.array(
                    [[2.316232e-001, 3.559933e+001, 1.435022e-277],
            [1.720466e-001, 1.770116e+001, 4.107648e-070],
            [2.001817e-001, 9.869938e+000, 5.619913e-023]]
                ),
            },
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lamols, np.array([[0.345118],
            [0.369596],
            [0.336918]]), RTOL
        )
        np.testing.assert_allclose(
            reg.lamsur, np.array([[0.296331],
            [0.309441],
            [0.300587]]), RTOL
        )
        np.testing.assert_allclose(
            reg.corr,
            np.array(
            [[1.      , 0.250537, 0.083957],
            [0.250537, 1.      , 0.251616],
            [0.083957, 0.251616, 1.      ]]
            ),
            RTOL,
        )
        np.testing.assert_allclose(reg.llik, -13737.409249)
        np.testing.assert_allclose(reg.errllik, -13735.053285)
        np.testing.assert_allclose(reg.surerrllik, -13647.929184)
        np.testing.assert_allclose(
            reg.lrtest, [174.24820135795017, 3, 1.5398532324238953e-37]
        )


class Test_SUR_error_gm(unittest.TestCase):
    def setUp(self):
        nat = libpysal.examples.load_example('NCOVR')
        self.db = gpd.read_file(nat.get_path("NAT.shp"))
        self.w = libpysal.weights.Queen.from_dataframe(self.db)
        self.w.transform = 'r'

    def test_error_gm(self):  # 2 equations
        y_var0 = ["HR80", "HR90"]
        x_var0 = [["PS80", "UE80"], ["PS90", "UE90"]]
        bigy0, bigX0, bigyvars0, bigXvars0 = sur_dictxy(self.db, y_var0, x_var0)
        reg = SURerrorGM(
            bigy0,
            bigX0,
            self.w,
            name_bigy=bigyvars0,
            name_bigX=bigXvars0,
            spat_diag=False,
            name_w="natqueen",
            name_ds="nat",
            nonspat_diag=True,
        )

        dict_compare(
            reg.bSUR,
            {
                0: np.array([[3.9774686], [0.8902122], [0.43050364]]),
                1: np.array([[2.93679118], [1.11002827], [0.48761542]]),
            },
            RTOL,
        )
        dict_compare(
            reg.sur_inf,
            {
                0: np.array(
                    [
                        [3.72514769e-01, 1.06773447e01, 1.29935073e-26],
                        [1.42242969e-01, 6.25839157e00, 3.88968202e-10],
                        [4.32238809e-02, 9.95985619e00, 2.28392844e-23],
                    ]
                ),
                1: np.array(
                    [
                        [3.36949019e-01, 8.71583239e00, 2.88630055e-18],
                        [1.34136264e-01, 8.27537784e00, 1.28048921e-16],
                        [4.03310502e-02, 1.20903229e01, 1.18818750e-33],
                    ]
                ),
            },
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lamsur, np.array([[0.55099267], [0.52364925]]), RTOL, ATOL
        )
        np.testing.assert_allclose(
            reg.corr, np.array([[1.0, 0.29038532], [0.29038532, 1.0]]), RTOL
        )
        np.testing.assert_allclose(
            reg.surchow,
            np.array(
                [
                    [5.51329, 1.0, 0.018873],
                    [1.775379, 1.0, 0.182718],
                    [1.1408, 1.0, 0.285483],
                ]
            ),
            rtol=RTOL,
            atol=ATOL,
        )

    def test_error_3eq_gm(self):  # Three equation example, unequal K
        y_var1 = ["HR60", "HR70", "HR80"]
        x_var1 = [["RD60", "PS60"], ["RD70", "PS70", "UE70"], ["RD80", "PS80"]]
        bigy1, bigX1, bigyvars1, bigXvars1 = sur_dictxy(self.db, y_var1, x_var1)
        reg = SURerrorGM(
            bigy1,
            bigX1,
            self.w,
            name_bigy=bigyvars1,
            name_bigX=bigXvars1,
            name_w="natqueen",
            name_ds="natregimes",
        )

        dict_compare(
            reg.bSUR,
            {
                0: np.array([[4.46897583], [2.15287009], [0.5979781]]),
                1: np.array([[7.10380031], [3.44965826], [1.10254808], [-0.15962263]]),
                2: np.array([[6.91299706], [3.70234954], [1.40532701]]),
            },
            RTOL,
        )
        dict_compare(
            reg.sur_inf,
            {
                0: np.array(
                    [
                        [1.44081634e-001, 3.10169709e001, 3.18308523e-211],
                        [1.25725320e-001, 1.71236000e001, 9.89616102e-066],
                        [1.11848242e-001, 5.34633439e000, 8.97533244e-008],
                    ]
                ),
                1: np.array(
                    [
                        [3.08054448e-001, 2.30602101e001, 1.16187890e-117],
                        [1.54010409e-001, 2.23988643e001, 4.03738963e-111],
                        [1.37435180e-001, 8.02231335e000, 1.03772013e-015],
                        [5.51073953e-002, -2.89657361e000, 3.77262126e-003],
                    ]
                ),
                2: np.array(
                    [
                        [1.60807064e-001, 4.29893867e001, 0.00000000e000],
                        [1.27136514e-001, 2.91210559e001, 1.94342017e-186],
                        [1.21987743e-001, 1.15202312e001, 1.04330705e-030],
                    ]
                ),
            },
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            reg.lamsur, np.array([[0.40589647], [0.42900222], [0.41682256]]), RTOL, ATOL
        )
        np.testing.assert_allclose(
            reg.corr,
            np.array(
                [
                    [1.0, 0.22987815, 0.13516187],
                    [0.22987815, 1.0, 0.2492023],
                    [0.13516187, 0.2492023, 1.0],
                ]
            ),
            RTOL,
        )


if __name__ == "__main__":
    unittest.main()
