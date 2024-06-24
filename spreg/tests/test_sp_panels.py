import unittest
import numpy as np
import libpysal
from libpysal.common import RTOL
from spreg.sp_panels import *
ATOL = 1e-12


class Test_GM_KKP(unittest.TestCase):
    def setUp(self):
        nat = libpysal.examples.load_example('NCOVR')
        self.db = libpysal.io.open(nat.get_path("NAT.dbf"),'r')
        self.w = libpysal.weights.Queen.from_shapefile(libpysal.examples.get_path("NAT.shp"))
        self.w.transform = 'r'
        self.y_var0 = ['HR70','HR80','HR90']
        self.x_var0 = ['RD70','RD80','RD90','PS70','PS80','PS90']
        self.y = np.array([self.db.by_col(name) for name in self.y_var0]).T
        self.x = np.array([self.db.by_col(name) for name in self.x_var0]).T


    def test_wide_ident(self): 
        reg = GM_KKP(self.y,self.x,self.w,full_weights=False,name_y=self.y_var0, name_x=self.x_var0)
        np.testing.assert_allclose(reg.betas,np.array([[ 6.49221562],
 [ 3.62445753],
 [ 1.31187779],
 [ 0.41777589],
 [22.81908224],
 [39.90993228]]),RTOL)
        np.testing.assert_allclose(reg.vm,np.array([[ 1.26948117e-02, -1.98160325e-06,  7.38157674e-05],
 [-1.98160325e-06,  7.69961725e-03, 1.13099329e-03],
 [ 7.38157674e-05,  1.13099329e-03,  7.26783636e-03]]),RTOL*10)
        np.testing.assert_equal(reg.name_x,  ['CONSTANT', 'RD', 'PS', 'lambda', ' sigma2_v', 'sigma2_1'])
        np.testing.assert_equal(reg.name_y,  'HR')

    def test_wide_full(self): 
        reg = GM_KKP(self.y,self.x,self.w,full_weights=True)

        np.testing.assert_allclose(reg.betas,np.array([[ 6.49193589],
 [ 3.55740165],
 [ 1.29462748],
 [ 0.4263399 ],
 [22.47241979],
 [45.82593532]]),RTOL)
        np.testing.assert_allclose(reg.vm,np.array([[ 1.45113773e-02, -2.14882672e-06,  8.54997693e-05],
 [-2.14882672e-06,  8.41929187e-03,  1.24553497e-03],
 [ 8.54997693e-05,  1.24553497e-03,  8.12448812e-03]]),RTOL)

    def test_long_ident(self): 
        bigy = self.y.reshape((self.y.size,1),order="F")
        bigx = self.x[:,0:3].reshape((self.x.shape[0]*3,1),order='F')
        bigx = np.hstack((bigx,self.x[:,3:6].reshape((self.x.shape[0]*3,1),order='F')))
        reg = GM_KKP(bigy,bigx,self.w,full_weights=False,name_y=['HR'], name_x=['RD','PS'])

        np.testing.assert_allclose(reg.betas,np.array([[ 6.49221562],
 [ 3.62445753],
 [ 1.31187779],
 [ 0.41777589],
 [22.81908224],
 [39.90993228]]),RTOL)
        np.testing.assert_allclose(reg.vm,np.array([[ 1.26948117e-02, -1.98160325e-06,  7.38157674e-05],
 [-1.98160325e-06,  7.69961725e-03, 1.13099329e-03],
 [ 7.38157674e-05,  1.13099329e-03,  7.26783636e-03]]),RTOL*10)
        np.testing.assert_equal(reg.name_x,  ['CONSTANT', 'RD', 'PS', 'lambda', ' sigma2_v', 'sigma2_1'])
        np.testing.assert_equal(reg.name_y,  'HR')

    def test_regimes(self):
        regimes = self.db.by_col("SOUTH")
        reg = GM_KKP(self.y,self.x,self.w,full_weights=False,regimes=regimes,
        name_y=self.y_var0, name_x=self.x_var0)
        np.testing.assert_allclose(reg.betas,np.array([[ 5.25856482],
 [ 3.19249165],
 [ 1.0056967 ],
 [ 7.94560642],
 [ 3.13931041],
 [ 1.53700634],
 [ 0.35979407],
 [22.5650005 ],
 [39.71516708]]),RTOL)
        np.testing.assert_allclose(np.sqrt(reg.vm.diagonal()),np.array([0.158986, 0.157543, 0.104128, 0.165254, 0.117737, 0.136666]),RTOL)
        np.testing.assert_equal(reg.name_x,  ['0_CONSTANT', '0_RD', '0_PS', '1_CONSTANT', '1_RD', '1_PS', 'lambda', ' sigma2_v', 'sigma2_1'])
        np.testing.assert_equal(reg.name_y,  'HR')
        np.testing.assert_allclose(reg.chow.regi,np.array([[1.420430e+02, 9.516507e-33],
       [7.311490e-02, 7.868543e-01],
       [9.652492e+00, 1.890949e-03]]),RTOL*10)
        np.testing.assert_allclose(reg.chow.joint[0],158.7225,RTOL)

if __name__ == '__main__':
    unittest.main()

