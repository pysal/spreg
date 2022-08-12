"""
Proposed spatial error model interface
"""

import numpy as np
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu as SuperLU
from .utils import set_warn
from . import diagnostics as DIAG
from . import user_output as USER
from . import summary_output as SUMMARY
from .w_utils import symmetrize
try:
    from scipy.optimize import minimize_scalar
    minimize_scalar_available = True
except ImportError:
    minimize_scalar_available = False
from .sputils import spdot, spfill_diagonal, spinv
from libpysal import weights
from .abstract_base import GenericModel


class Error(GenericModel):
    def __init__(self, X, y, w, vm=False, name_y=None, name_X=None,
                 name_w=None, name_ds=None):
        # Input checking
        n = USER.check_arrays(y, X)
        y = USER.check_y(y, n)
        USER.check_weights(w, y, w_required=True)
        x_constant, name_x, warn = USER.check_constant(X, name_X)
        set_warn(self, warn)

        self.name_ds = USER.set_name_ds(name_ds)
        self.name_y = USER.set_name_y(name_y)
        self.name_x = USER.set_name_x(name_X, X)
        self.name_x.append('lambda')
        self.name_w = USER.set_name_w(name_w, w)

        # Assign data attributes
        super().__init__(X, y)
        self.w = w

    def err_c_loglik(self, lam, n, y, ylag, x, xlag, W):
        # concentrated log-lik for error model, no constants, brute force
        ys = y - lam * ylag
        xs = x - lam * xlag
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        sig2 = ee[0][0] / n
        nlsig2 = (n / 2.0) * np.log(sig2)
        a = -lam * W
        np.fill_diagonal(a, 1.0)
        jacob = np.log(np.linalg.det(a))
        # this is the negative of the concentrated log lik for minimization
        clik = nlsig2 - jacob
        return clik

    def err_c_loglik_sp(self, lam, n, y, ylag, x, xlag, I, Wsp):
        # concentrated log-lik for error model, no constants, LU
        if isinstance(lam, np.ndarray):
            if lam.shape == (1, 1):
                lam = lam[0][0]  # why does the interior value change?
        ys = y - lam * ylag
        xs = x - lam * xlag
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        sig2 = ee[0][0] / n
        nlsig2 = (n / 2.0) * np.log(sig2)
        a = I - lam * Wsp
        LU = SuperLU(a.tocsc())
        jacob = np.sum(np.log(np.abs(LU.U.diagonal())))
        # this is the negative of the concentrated log lik for minimization
        clik = nlsig2 - jacob
        return clik

    def err_c_loglik_ord(self, lam, n, y, ylag, x, xlag, evals):
        # concentrated log-lik for error model, no constants, eigenvalues
        ys = y - lam * ylag
        xs = x - lam * xlag
        ysys = np.dot(ys.T, ys)
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        x1 = np.dot(xsxsi, xsys)
        x2 = np.dot(xsys.T, x1)
        ee = ysys - x2
        sig2 = ee[0][0] / n
        nlsig2 = (n / 2.0) * np.log(sig2)
        revals = lam * evals
        jacob = np.log(1 - revals).sum()
        if isinstance(jacob, complex):
            jacob = jacob.real
        # this is the negative of the concentrated log lik for minimization
        clik = nlsig2 - jacob
        return clik

    def _fit_ml(self, method, epsilon):
        ylag = weights.lag_spatial(w, self.y)
        xlag = self.get_x_lag(w, regimes_att)

        # call minimizer using concentrated log-likelihood to get lambda
        if method in ['full', 'lu', 'ord']:
            if method == 'full':  
                W = w.full()[0]      # need dense here
                res = minimize_scalar(err_c_loglik, 0.0, bounds=(-1.0, 1.0),
                                      args=(self.n, self.y, ylag, self.x,
                                            xlag, W), method='bounded',
                                      tol=epsilon)
            elif method == 'lu':
                I = sp.identity(w.n)
                Wsp = w.sparse   # need sparse here
                res = minimize_scalar(err_c_loglik_sp, 0.0, bounds=(-1.0,1.0),
                                      args=(self.n, self.y, ylag, 
                                            self.x, xlag, I, Wsp),
                                      method='bounded', tol=epsilon)
                W = Wsp
            elif method == 'ord':
                # check on symmetry structure
                if w.asymmetry(intrinsic=False) == []:
                    ww = symmetrize(w)
                    WW = np.array(ww.todense())
                    evals = la.eigvalsh(WW)
                    W = WW
                else:
                    W = w.full()[0]      # need dense here
                    evals = la.eigvals(W)
                res = minimize_scalar(
                    err_c_loglik_ord, 0.0, bounds=(-1.0, 1.0),
                    args=(self.n, self.y, ylag, self.x,
                          xlag, evals), method='bounded',
                    tol=epsilon)
        else:
            raise Exception("{0} is an unsupported method".format(method))

        self.lam = res.x

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - self.n / 2.0 * ln2pi - self.n / 2.0

        self.logll = llik

        # b, residuals and predicted values

        ys = self.y - self.lam * ylag
        xs = self.x - self.lam * xlag
        xsxs = np.dot(xs.T, xs)
        xsxsi = np.linalg.inv(xsxs)
        xsys = np.dot(xs.T, ys)
        b = np.dot(xsxsi, xsys)

        self.betas = np.vstack((b, self.lam))

        self.u = y - np.dot(self.x, b)
        self.predy = self.y - self.u

        # residual variance

        self.e_filtered = self.u - self.lam * weights.lag_spatial(w, self.u)
        self.sig2 = np.dot(self.e_filtered.T, self.e_filtered) / self.n

        # variance-covariance matrix betas

        varb = self.sig2 * xsxsi

        # variance-covariance matrix lambda, sigma

        a = -self.lam * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum()

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()

        v1 = np.vstack((tr2 + tr3,
                        tr1 / self.sig2))
        v2 = np.vstack((tr1 / self.sig2,
                        self.n / (2.0 * self.sig2 ** 2)))

        v = np.hstack((v1, v2))

        self.vm1 = np.linalg.inv(v)

        # create variance matrix for beta, lambda
        vv = np.hstack((varb, np.zeros((self.k, 1))))
        vv1 = np.hstack(
            (np.zeros((1, self.k)), self.vm1[0, 0] * np.ones((1, 1))))

        self.vm = np.vstack((vv, vv1))
    def fit(self, method="gm", epsilon=1e-7):
        method = method.lower()
        if method in ["full", "lu", "ord"]:
            self._fit_ml(method, epsilon)
            self.ml = True
        elif method == "gm":
            self._fit_gm()
            self.ml = False
        else:
            # Crash gracefully
            print(f"{method} is an unsupported method")
            return
        return self

    def summary(self):
        self.aic = DIAG.akaike(reg=self)
        self.schwarz = DIAG.schwarz(reg=self)
        SUMMARY.ML_Error(reg=self, w=self.w, vm=self.vm, spat_diag=False)
