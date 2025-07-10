import unittest
import libpysal
from libpysal import weights
import numpy as np
import spreg
from spreg import error_sp as SP
from libpysal.common import RTOL
import pandas as pd
from libpysal.examples import load_example
from spreg import GMM_Error_Regimes

class TestBaseGMError(unittest.TestCase):
    def setUp(self):
        db=libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = SP.BaseGM_Error(self.y, self.X, self.w.sparse)
        betas = np.array([[ 47.94371455], [  0.70598088], [ -0.55571746], [  0.37230161]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.4739775])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([ 31.89620319])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 191.73716465732355
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

class TestGMError(unittest.TestCase):
    def setUp(self):
        db=libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = SP.GM_Error(self.y, self.X, self.w)
        betas = np.array([[ 47.94371455], [  0.70598088], [ -0.55571746], [  0.37230161]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 27.4739775])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        e = np.array([ 31.89620319])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 52.9930255])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 191.73716465732355
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3495097406012179
        np.testing.assert_allclose(reg.pr2,pr2)
        std_err = np.array([ 12.32416094,   0.4989716 ,   0.1785863 ])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  3.89022140e+00,   1.00152805e-04], [  1.41487186e+00,   1.57106070e-01], [ -3.11175868e+00,   1.85976455e-03]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

class TestBaseGMEndogError(unittest.TestCase):
    def setUp(self):
        db=libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        self.X = np.hstack((np.ones(self.y.shape),self.X))
        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = SP.BaseGM_Endog_Error(self.y, self.X, self.yd, self.q, self.w.sparse)
        betas = np.array([[ 55.36095292], [  0.46411479], [ -0.66883535], [  0.38989939]])
        #raising a warning causes a failure...
        print('Running reduced precision test in L120 of test_error_sp.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL+.0001)
        u = np.array([ 26.55951566])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 31.23925425])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 53.9074875])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        #std_y
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        #vm
        vm = np.array([[  5.29158422e+02,  -1.57833675e+01,  -8.38021080e+00],
       [ -1.57833675e+01,   5.40235041e-01,   2.31120327e-01],
       [ -8.38021080e+00,   2.31120327e-01,   1.44977385e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 192.50022721929574
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

class TestGMEndogError(unittest.TestCase):
    def setUp(self):
        db=libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        self.X = np.array(X).T
        yd = []
        yd.append(db.by_col("CRIME"))
        self.yd = np.array(yd).T
        q = []
        q.append(db.by_col("DISCBD"))
        self.q = np.array(q).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = SP.GM_Endog_Error(self.y, self.X, self.yd, self.q, self.w)
        betas = np.array([[ 55.36095292], [  0.46411479], [ -0.66883535], [  0.38989939]])
        print('Running reduced precision test in L175 of test_error_sp.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL +.0001)
        u = np.array([ 26.55951566])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e = np.array([ 31.23925425])
        np.testing.assert_allclose(reg.e_filtered[0],e,RTOL)
        predy = np.array([ 53.9074875])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 3
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.   ,  19.531])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my,RTOL)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy,RTOL)
        vm = np.array([[  5.29158422e+02,  -1.57833675e+01,  -8.38021080e+00],
       [ -1.57833675e+01,   5.40235041e-01,   2.31120327e-01],
       [ -8.38021080e+00,   2.31120327e-01,   1.44977385e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        pr2 = 0.346472557570858
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        sig2 = 192.50022721929574
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        std_err = np.array([ 23.003401  ,   0.73500657,   0.38075777])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[ 2.40664208,  0.01609994], [ 0.63144305,  0.52775088], [-1.75659016,  0.07898769]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

    def test_model_1(self):
        reg = SP.GM_Endog_Error(self.y, self.X, self.yd, self.q, self.w, slx_lags=1)
        betas = np.array([[76.77491588], [0.21332977], [-0.62211115], [-0.91101459], [0.40667462]])
        np.testing.assert_allclose(reg.betas,betas,RTOL +.0001)
        print('Running reduced precision test in L205 of test_error_sp.py')
        np.testing.assert_allclose(reg.betas,betas,RTOL +.0001)
        u = np.array([25.419675300942714])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([31.207183005033297])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        predy = np.array([55.04732769905729])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 4
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([1.0, 19.531, 18.594])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([15.72598])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([1.0, 19.531, 18.594, 15.72598])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        mean_y = 38.43622446938775
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array( [[1241.3679785142479, -25.45186633997482, -21.57345029999624, -15.508445428953587],
                         [-25.451866339974888, 0.6839492993049352, 0.2740158770507004, 0.3315209826545654], 
                         [-21.573450299996, 0.27401587705069513, 0.6877711470567254, 0.20979319854002507], 
                         [-15.508445428953618, 0.3315209826545651, 0.2097931985400285, 0.2178378881170716]] )
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 203.39160138851173
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)     

class TestGMCombo(unittest.TestCase):
    def setUp(self):
        db=libpysal.io.open(libpysal.examples.get_path("columbus.dbf"),"r")
        y = np.array(db.by_col("HOVAL"))
        self.y = np.reshape(y, (49,1))
        X = []
        X.append(db.by_col("INC"))
        X.append(db.by_col("CRIME"))
        self.X = np.array(X).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        # Only spatial lag
        reg = SP.GM_Combo(self.y, self.X, w=self.w)
        e_reduced = np.array([ 28.18617481])
        np.testing.assert_allclose(reg.e_pred[0],e_reduced,RTOL)
        predy_e = np.array([ 52.28082782])
        np.testing.assert_allclose(reg.predy_e[0],predy_e,RTOL)
        betas = np.array([[ 57.61123515],[  0.73441313], [ -0.59459416], [ -0.21762921], [  0.54732051]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        u = np.array([ 25.57932637])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([ 31.65374945])
        np.testing.assert_allclose(reg.e_filtered[0],e_filtered,RTOL)
        predy = np.array([ 54.88767685])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 4
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([ 80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([  1.     ,  19.531  ,  15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([  35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([  1.       ,  19.531    ,  15.72598  ,  35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        my = 38.43622446938776
        np.testing.assert_allclose(reg.mean_y,my)
        sy = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,sy)
        vm = np.array([  5.22438333e+02,   2.38012875e-01,   3.20924173e-02,
         2.15753579e-01])
        np.testing.assert_allclose(np.diag(reg.vm),vm,RTOL)
        sig2 = 181.78650186468832
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)
        pr2 = 0.3018280166937799
        np.testing.assert_allclose(reg.pr2,pr2,RTOL)
        pr2_e = 0.3561355586759414
        np.testing.assert_allclose(reg.pr2_e,pr2_e,RTOL)
        std_err = np.array([ 22.85692222,  0.48786559,  0.17914356,  0.46449318])
        np.testing.assert_allclose(reg.std_err,std_err,RTOL)
        z_stat = np.array([[  2.52051597e+00,   1.17182922e-02], [  1.50535954e+00,   1.32231664e-01], [ -3.31909311e+00,   9.03103123e-04], [ -4.68530506e-01,   6.39405261e-01]])
        np.testing.assert_allclose(reg.z_stat,z_stat,RTOL)

    def test_model_1(self):
        # SLX > 0, slx_vars = "All" and  name_x=["income", "crime"]
        reg = SP.GM_Combo(self.y, self.X, w=self.w,slx_lags=1, slx_vars="All", name_x=["income", "crime"])
        betas = np.array([[-1.68549776], [1.023858], [-0.60027516], [-1.24056691], [0.58796177], [1.1429901], [-0.64301075]])
        np.testing.assert_allclose(reg.betas,betas,RTOL+.0001)
        u = np.array([39.60278686988822])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([27.387967128789217])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        predy = np.array([40.864216130111785])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 6
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([1.0, 19.531, 15.72598, 18.594, 24.7142675])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend = np.array([35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([1.0, 19.531, 15.72598, 18.594, 24.7142675, 35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        mean_y = 38.43622446938775
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array( [[310.6283901851071, -1.8625762732506532, 0.23637750381058148, 
                         -2.490257967992913, -3.540727963871908, -3.3853342129346014],
                           [-1.862576273250409, 0.25454991581277797, 0.04775611402222607,
                             -0.3159417698794814, -0.0361064859627287, 0.0622148895979572], 
                             [0.23637750381059774, 0.04775611402223329, 0.0348942850522157,
                               -0.08708418052902171, -0.04146284919474196, 0.01503172670815147],
                                 [-2.4902579679940353, -0.31594176987947975, -0.08708418052899158,
                                   1.1272379141717317, 0.17905841078689427, -0.32987287666815707],
                                     [-3.5407279638721842, -0.03610648596273606, -0.04146284919474014,
                                       0.1790584107869242, 0.0916798380512302, -0.00838180184803425], 
                                       [-3.3853342129346173, 0.062214889597958206, 0.015031726708141036,
                                         -0.32987287666815596, -0.00838180184802332, 0.18434843597847317]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 136.12652143339153
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

    def test_model_2(self):
        # slx_vars = [True, False]
        reg = SP.GM_Combo(self.y, self.X, w=self.w,slx_vars=[True, False])
        betas = np.array([ [57.6112352], [0.73441313], [-0.59459416], [-0.21762921], [0.54732051] ])
        np.testing.assert_allclose(reg.betas,betas,RTOL+.0001)
        u = np.array([25.57932638895275])
        np.testing.assert_allclose(reg.u[0],u,RTOL)
        e_filtered = np.array([31.6537488410598])
        np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
        predy = np.array([54.887676611047254])
        np.testing.assert_allclose(reg.predy[0],predy,RTOL)
        n = 49
        np.testing.assert_allclose(reg.n,n,RTOL)
        k = 4
        np.testing.assert_allclose(reg.k,k,RTOL)
        y = np.array([80.467003])
        np.testing.assert_allclose(reg.y[0],y,RTOL)
        x = np.array([1.0, 19.531, 15.72598])
        np.testing.assert_allclose(reg.x[0],x,RTOL)
        yend =yend = np.array([35.4585005])
        np.testing.assert_allclose(reg.yend[0],yend,RTOL)
        z = np.array([1.0, 19.531, 15.72598, 35.4585005])
        np.testing.assert_allclose(reg.z[0],z,RTOL)
        mean_y = 38.43622446938775
        np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
        std_y = 18.466069465206047
        np.testing.assert_allclose(reg.std_y,std_y,RTOL)
        vm = np.array( [[522.4388491854916, -6.07257215716109, -1.9142891285650294, -8.971342288524905],
                         [-6.072572157160922, 0.23801283897827286, 0.04701607746122861, 0.028096399569782167],
                           [-1.914289128564947, 0.04701607746122825, 0.03209241537273229, 0.0031496871402078414], 
                           [-8.971342288524905, 0.02809639956978561, 0.003149687140209358, 0.21575388784831606]] )
        np.testing.assert_allclose(reg.vm,vm,RTOL)
        sig2 = 181.78650523727262
        np.testing.assert_allclose(reg.sig2,sig2,RTOL)

class TestGMMError(unittest.TestCase):
    def setUp(self):
        try:
            self.db = pd.read_csv(libpysal.examples.get_path('columbus.csv'))
        except ValueError:
            import geopandas as gpd
            self.db = gpd.read_file(libpysal.examples.get_path('columbus.dbf'))
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, estimator='kp98') #GM_Error
        betas = np.array([[ 47.94371455], [  0.70598088], [ -0.55571746], [  0.37230161]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  1.51884943e+02,  -5.37622793e+00,  -1.86970286e+00], [ -5.37622793e+00,   2.48972661e-01,   5.26564244e-02], [ -1.86970286e+00,   5.26564244e-02, 3.18930650e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC"]], yend=self.db[["CRIME"]], q=self.db[["DISCBD"]], w=self.w, estimator='kp98') #GM_Endog_Error
        betas = np.array([[ 55.36095292], [  0.46411479], [ -0.66883535], [  0.38989939]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  5.29158422e+02,  -1.57833675e+01,  -8.38021080e+00],
       [ -1.57833675e+01,   5.40235041e-01,   2.31120327e-01],
       [ -8.38021080e+00,   2.31120327e-01,   1.44977385e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, estimator='kp98', add_wy=True) #GM_Combo
        betas = np.array([[ 57.61123515],[  0.73441313], [ -0.59459416], [ -0.21762921], [  0.54732051]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([  5.22438333e+02,   2.38012875e-01,   3.20924173e-02,
         2.15753579e-01])
        np.testing.assert_allclose(np.diag(reg.vm),vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, estimator='kp98', slx_lags=1) #SLX_error
        betas = np.array([[29.46053861],
       [ 0.82091985],
       [-0.57543046],
       [ 0.47808558],
       [ 0.30069346],
       [ 0.35833037]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[ 1.01097138e+03, -9.41982352e+00, -2.05551111e+00,
        -2.90663546e+01, -1.04492128e+01],
       [-9.41982352e+00,  2.70128986e-01,  4.98208129e-02,
         1.18829756e-01,  5.97170349e-02],
       [-2.05551111e+00,  4.98208129e-02,  3.45149007e-02,
         2.31378455e-02, -6.15257324e-03],
       [-2.90663546e+01,  1.18829756e-01,  2.31378455e-02,
         1.06830398e+00,  3.06585506e-01],
       [-1.04492128e+01,  5.97170349e-02, -6.15257324e-03,
         3.06585506e-01,  1.51494962e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, estimator='hom', A1='hom_sc') #GM_Error_Hom
        betas = np.array([[ 47.9478524 ], [  0.70633223], [ -0.55595633], [  0.41288558]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  1.51340717e+02,  -5.29057506e+00,  -1.85654540e+00, -2.39139054e-03], [ -5.29057506e+00,   2.46669610e-01, 5.14259101e-02, 3.19241302e-04], [ -1.85654540e+00,   5.14259101e-02, 3.20510550e-02,  -5.95640240e-05], [ -2.39139054e-03,   3.19241302e-04, -5.95640240e-05,  3.36690159e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC"]], yend=self.db[["CRIME"]], q=self.db[["DISCBD"]], w=self.w, estimator='hom', A1='hom_sc') #GM_Endog_Error_Hom
        betas = np.array([[ 55.36575166], [  0.46432416], [ -0.66904404], [  0.43205526]])
        vm = np.array([[  5.52064057e+02,  -1.61264555e+01,  -8.86360735e+00, 1.04251912e+00], [ -1.61264555e+01,   5.44898242e-01, 2.39518645e-01, -1.88092950e-02], [ -8.86360735e+00,   2.39518645e-01, 1.55501840e-01, -2.18638648e-02], [  1.04251912e+00, -1.88092950e-02, -2.18638648e-02, 3.71222222e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC"]], w=self.w, estimator='hom', add_wy=True, A1='hom_sc') #GM_Combo_Hom
        betas = np.array([[ 10.12541428], [  1.56832263], [  0.15132076], [  0.21033397]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  2.33694742e+02,  -6.66856869e-01,  -5.58304254e+00, 4.85488380e+00], [ -6.66856869e-01,   1.94241504e-01, -5.42327138e-02, 5.37225570e-02], [ -5.58304254e+00,  -5.42327138e-02, 1.63860721e-01, -1.44425498e-01], [  4.85488380e+00, 5.37225570e-02, -1.44425498e-01, 1.78622255e-01]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, estimator='hom', slx_lags=1) #SLX_error_Hom
        betas = np.array([[29.45631607],
       [ 0.82147165],
       [-0.57539916],
       [ 0.47867457],
       [ 0.30033727],
       [ 0.40129812]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[ 1.06448125e+03, -1.00662342e+01, -2.27271870e+00,
        -3.04900044e+01, -1.08579426e+01, -1.04655994e-02],
       [-1.00662342e+01,  2.72517164e-01,  5.08167383e-02,
         1.44806006e-01,  6.54774010e-02, -2.75387247e-04],
       [-2.27271870e+00,  5.08167383e-02,  3.44272896e-02,
         2.96786313e-02, -3.01082586e-03,  5.42396309e-05],
       [-3.04900044e+01,  1.44806006e-01,  2.96786313e-02,
         1.10589204e+00,  3.13859279e-01,  3.30913956e-04],
       [-1.08579426e+01,  6.54774010e-02, -3.01082586e-03,
         3.13859279e-01,  1.54648513e-01,  2.16465817e-04],
       [-1.04655994e-02, -2.75387247e-04,  5.42396309e-05,
         3.30913956e-04,  2.16465817e-04,  3.91478879e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, step1c=True) #GM_Error_Het
        betas = np.array([[ 47.99626638], [  0.71048989], [ -0.55876126], [  0.41178776]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  1.31767529e+02,  -3.58368748e+00,  -1.65090647e+00,
              0.00000000e+00],
           [ -3.58368748e+00,   1.35513711e-01,   3.77539055e-02,
              0.00000000e+00],
           [ -1.65090647e+00,   3.77539055e-02,   2.61042702e-02,
              0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.82398517e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC"]], yend=self.db[["CRIME"]], q=self.db[["DISCBD"]], w=self.w, step1c=True) #GM_Endog_Error_Het
        betas = np.array([[ 55.39707924], [  0.46563046], [ -0.67038326], [  0.41135023]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  8.34637805e+02,  -2.16932259e+01,  -1.33327894e+01,
                  1.65840848e+00],
               [ -2.16932259e+01,   5.97683070e-01,   3.39503523e-01,
                 -3.90111107e-02],
               [ -1.33327894e+01,   3.39503523e-01,   2.19008080e-01,
                 -2.81929695e-02],
               [  1.65840848e+00,  -3.90111107e-02,  -2.81929695e-02,
                  3.15686105e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, add_wy=True, step1c=True) #GM_Combo_Het
        betas = np.array([[ 57.7778574 ], [  0.73034922], [ -0.59257362], [ -0.2230231 ], [  0.56636724]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[  4.86218274e+02,  -2.77268729e+00,  -1.59987770e+00,
             -1.01969471e+01,   2.74302006e+00],
           [ -2.77268729e+00,   1.04680972e-01,   2.51172238e-02,
              1.95136385e-03,   3.70052723e-03],
           [ -1.59987770e+00,   2.51172238e-02,   2.15655720e-02,
              7.65868344e-03,  -7.30173070e-03],
           [ -1.01969471e+01,   1.95136385e-03,   7.65868344e-03,
              2.78273684e-01,  -6.89402590e-02],
           [  2.74302006e+00,   3.70052723e-03,  -7.30173070e-03,
             -6.89402590e-02,   7.12034037e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL*10)

        reg = SP.GMM_Error(self.db[["HOVAL"]], self.db[["INC", "CRIME"]], w=self.w, slx_lags=1) #SLX_error_Het
        betas = np.array([[29.38238574],[ 0.82921502],[-0.57499819],[ 0.48748671],[ 0.29556428],[ 0.39636619]])
        np.testing.assert_allclose(reg.betas,betas,RTOL)
        vm = np.array([[ 5.97495766e+02, -5.55463269e+00, -6.81492201e-01,
                        -1.53802421e+01, -7.21116007e+00,  0.00000000e+00],
                        [-5.55463269e+00,  1.34368489e-01,  2.10846256e-02,
                          3.83654841e-02,  5.54102390e-02,  0.00000000e+00],
                          [-6.81492201e-01,  2.10846256e-02,  2.56742519e-02,
                            2.40939941e-03, -1.84082192e-02,  0.00000000e+00],
                            [-1.53802421e+01,  3.83654841e-02,  2.40939941e-03,
                             6.69703341e-01,  1.58284990e-01,  0.00000000e+00],
                             [-7.21116007e+00,  5.54102390e-02, -1.84082192e-02,
                              1.58284990e-01,  1.30057684e-01,  0.00000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                               0.00000000e+00,  0.00000000e+00,  2.67800026e-02]])
        np.testing.assert_allclose(reg.vm,vm,RTOL)

    def setUp_regimes(self): 
        nat = load_example('Natregimes')
        db = libpysal.io.open(nat.get_path("natregimes.dbf"),'r')
        self.y_var = 'HR90'
        self.y = np.array([db.by_col(self.y_var)]).reshape(3085,1)
        self.x_var = ['PS90','UE90']
        self.x = np.array([db.by_col(name) for name in self.x_var]).T
        self.r_var = 'SOUTH'
        self.regimes = db.by_col(self.r_var)
        self.w = libpysal.weights.Rook.from_shapefile(nat.get_path("natregimes.shp"))
        self.w.transform = 'r' 
        
    def test_regimes(self):
        self.setUp_regimes()
        reg = GMM_Error_Regimes(self.y, self.x, self.regimes, self.w)
        betas = np.array([
        [0.068987],
        [0.788486], 
        [0.539782], 
        [5.094800],  
        [1.196472],  
        [0.601786],  
        [0.427196]  ])
        np.testing.assert_allclose(reg.betas,betas,rtol=1e-4, atol=1e-5)
        vm = np.array([
        [1.26869e-01, -2.58360e-02, -1.86910e-02, 3.49500e-03, -9.33000e-04, -4.78000e-04, 0.00000e+00],
        [-2.58360e-02, 4.91320e-02, 2.88600e-03, -8.58000e-04, 1.86800e-03, 1.53000e-04, 0.00000e+00],
        [-1.86910e-02, 2.88600e-03, 3.53200e-03, -1.59000e-04, 2.13000e-04, 6.40000e-05, 0.00000e+00],
        [3.49500e-03, -8.58000e-04, -1.59000e-04, 4.70141e-01, -9.18920e-02, -5.77590e-02, 0.00000e+00],
        [-9.33000e-04, 1.86800e-03, 2.13000e-04, -9.18920e-02, 1.28565e-01, 1.05540e-02, 0.00000e+00],
        [-4.78000e-04, 1.53000e-04, 6.40000e-05, -5.77590e-02, 1.05540e-02, 8.58300e-03, 0.00000e+00],
        [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.37000e-04]])
        np.testing.assert_allclose(reg.vm,vm,rtol=1e-4, atol=1e-5)

class TestGMComboExample(unittest.TestCase):
    def setUp(self):
        self.db = libpysal.io.open(libpysal.examples.get_path('columbus.dbf'), 'r')
        self.y = np.array(self.db.by_col("HOVAL"))
        self.y = np.reshape(self.y, (49,1))
        self.X = np.array([self.db.by_col("INC")]).T
        self.yd = np.array([self.db.by_col("CRIME")]).T
        self.q = np.array([self.db.by_col("DISCBD")]).T
        self.w = libpysal.weights.Rook.from_shapefile(libpysal.examples.get_path("columbus.shp"))
        self.w.transform = 'r'

    def test_model(self):
        import io
        import sys
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            reg = SP.GM_Combo(self.y, self.X, self.yd, self.q, w=self.w, 
                       name_x=['inc'], name_y='hoval', name_yend=['crime'], name_q=['discbd'],
                       name_ds='columbus', vm=True)
            betas = np.array([[111.90452543], [-0.31201292], [-1.36030107], [-0.5319876], [0.68161358]])
            np.testing.assert_allclose(reg.betas,betas,RTOL+.0001)
            u = np.array([14.911951654904712])
            np.testing.assert_allclose(reg.u[0],u,RTOL)
            e_filtered = np.array([23.46232931535747])
            np.testing.assert_allclose(reg.e_filtered[0], e_filtered, RTOL)
            predy = np.array([65.5550513450953])
            np.testing.assert_allclose(reg.predy[0],predy,RTOL)
            n = 49
            np.testing.assert_allclose(reg.n,n,RTOL)
            k = 4
            np.testing.assert_allclose(reg.k,k,RTOL)
            y = np.array([80.467003])
            np.testing.assert_allclose(reg.y[0],y,RTOL)
            x = np.array([1.0, 19.531])
            np.testing.assert_allclose(reg.x[0],x,RTOL)
            yend = np.array([15.72598, 35.4585005])
            np.testing.assert_allclose(reg.yend[0],yend,RTOL)
            z = np.array([1.0, 19.531, 15.72598, 35.4585005 ])
            np.testing.assert_allclose(reg.z[0],z,RTOL)
            mean_y = 38.43622446938775
            np.testing.assert_allclose(reg.mean_y,mean_y,RTOL)
            std_y = 18.466069465206047
            np.testing.assert_allclose(reg.std_y,std_y,RTOL)
            vm = np.array( [[5016.050352148409, -76.53478080595559, -44.251807673048404, -59.28554254165969],
                         [-76.53478080595606, 1.4067533165464567, 0.7448544825341058, 0.7791558312477468],
                           [-44.25180767304751, 0.7448544825340884, 0.48156504963672947, 0.42796705603821317], 
                           [-59.285542541663446, 0.779155831247796, 0.427967056038256, 0.8430846436101599]] )
            np.testing.assert_allclose(reg.vm,vm,RTOL)
            print(reg.output)
            print(reg.summary)
        output = f.getvalue()

if __name__ == '__main__':
    unittest.main()
