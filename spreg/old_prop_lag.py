"""
Proposed spatial lag model interface, combining ML and GM
"""

# Common imports
import numpy as np
from .import user_output as USER
from . import summary_output as SUMMARY
from .utils import set_warn
from .abstract_base import GenericModel

# ML_Lag imports
import numpy.linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import splu as SuperLU
from .utils import inverse_prod
from .sputils import spdot, spfill_diagonal, spinv, sphstack
from . import diagnostics as DIAG
from .w_utils import symmetrize
from libpysal import weights
from scipy.optimize import minimize_scalar
from libpysal.weights.spatial_lag import lag_spatial

# GM_Lag imports
from .utils import set_endog, sp_att
from . import robust as ROBUST


class Lag(GenericModel):
    def __init__(self, X, y, w, name_X=None, name_y=None, vm_flag=False,
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
        name_ylag = USER.set_name_yend_sp(self.name_y)
        self.name_x.append(name_ylag)  # rho changed to last position
        self.name_w = USER.set_name_w(name_w, w)

        # Assign data attributes
        super().__init__(X, y)
        self.vm_flag = vm_flag  # if True include covariance matrix in results
        self.w = w

    def _model(self, params):
        return spdot(self.X, params[1:-1]) + params[-1] * lag_spatial(self.w, self.y) + params[0]

    def _objective(self, method, rho, n, e0, e1, W=None, eye=None, Wsp=None, evals=None):
        if method == "full":
            # concentrated log-lik for lag model, no constants, brute force
            er = e0 - rho * e1
            sig2 = spdot(er.T, er) / n
            nlsig2 = (n / 2.0) * np.log(sig2)
            a = -rho * W
            spfill_diagonal(a, 1.0)
            jacob = np.log(np.linalg.det(a))
            # this is the negative of the concentrated log lik for minimization
            clik = nlsig2 - jacob
            return clik
        elif method == "lu":
            # concentrated log-lik for lag model, sparse algebra
            if isinstance(rho, np.ndarray):
                if rho.shape == (1, 1):
                    rho = rho[0][0]  # why does the interior value change?
            er = e0 - rho * e1
            sig2 = spdot(er.T, er) / n
            nlsig2 = (n / 2.0) * np.log(sig2)
            a = eye - rho * Wsp
            LU = SuperLU(a.tocsc())
            jacob = np.sum(np.log(np.abs(LU.U.diagonal())))
            clike = nlsig2 - jacob
            return clike
        elif method == "ord":
            # concentrated log-lik for lag model, no constants, Ord eigenvalue method
            er = e0 - rho * e1
            sig2 = spdot(er.T, er) / n
            nlsig2 = (n / 2.0) * np.log(sig2)
            revals = rho * evals
            jacob = np.log(1 - revals).sum()
            if isinstance(jacob, complex):
                jacob = jacob.real
            # this is the negative of the concentrated log lik for minimization
            clik = nlsig2 - jacob
            return clik

    def fit(self, method="gm", yend=None, q=None, epsilon=1e-7):
        method = method.lower()
        if method in ["full", "lu", "ord"]:
            self._fit_ml(method, epsilon)
            self.ml = True
        elif method == "gm":
            self._fit_gm(yend=yend, q=q)
            self.ml = False
        else:
            # Crash gracefully
            print(f"{method} is an unsupported method")
            return
        return self

    def _fit_ml(self, method, epsilon):
        # Set up main regression variables and spatial filters
        ylag = weights.lag_spatial(self.w, self.y)
        xtx = spdot(self.X.T, self.X)
        xtxi = la.inv(xtx)
        xty = spdot(self.X.T, self.y)
        xtyl = spdot(self.X.T, ylag)
        b0 = spdot(xtxi, xty)
        b1 = spdot(xtxi, xtyl)
        e0 = self.y - spdot(self.X, b0)
        e1 = ylag - spdot(self.X, b1)

        if method == "full":
            W = self.w.full()[0]     # moved here
            res = minimize_scalar(self._objective, 0.0, bounds=(-1.0, 1.0),
                                  args=(method,
                                        self.n, e0, e1, W, None, None, None),
                                  method='bounded', tol=epsilon)
        elif method == "lu":
            eye = sp.identity(self.w.n)
            Wsp = self.w.sparse  # moved here
            W = Wsp
            res = minimize_scalar(self._objective, 0.0, bounds=(-1.0, 1.0),
                                  args=(method, self.n, e0, e1, None, eye, Wsp, None),
                                  method='bounded', tol=epsilon)
        elif method == "ord":
            # check on symmetry structure
            if self.w.asymmetry(intrinsic=False) == []:
                ww = symmetrize(self.w)
                WW = np.array(ww.todense())
                evals = la.eigvalsh(WW)
                W = WW
            else:
                W = self.w.full()[0]     # moved here
                evals = la.eigvals(W)
            res = minimize_scalar(self._objective, 0.0, bounds=(-1.0, 1.0),
                                  args=(method,
                                        self.n, e0, e1, None, None, None, evals),
                                  method='bounded', tol=epsilon)
        self.rho = res.x[0][0]

        # compute full log-likelihood, including constants
        ln2pi = np.log(2.0 * np.pi)
        llik = -res.fun - self.n / 2.0 * ln2pi - self.n / 2.0
        self.logll = llik[0][0]

        # b, residuals and predicted values

        b = b0 - self.rho * b1
        self.params = np.vstack((b, self.rho))   # rho added as last coefficient
        self.u = e0 - self.rho * e1
        self.predy = self.y - self.u

        xb = spdot(self.X, b)

        self.predy_e = inverse_prod(
            self.w.sparse, xb, self.rho, inv_method="power_exp", threshold=epsilon)
        self.e_pred = self.y - self.predy_e

        # residual variance
        self._cache = {}
        self.sig2 = self.sig2n  # no allowance for division by n-k

        # information matrix
        # if w should be kept sparse, how can we do the following:
        a = -self.rho * W
        spfill_diagonal(a, 1.0)
        ai = spinv(a)
        wai = spdot(W, ai)
        tr1 = wai.diagonal().sum()  # same for sparse and dense

        wai2 = spdot(wai, wai)
        tr2 = wai2.diagonal().sum()

        waiTwai = spdot(wai.T, wai)
        tr3 = waiTwai.diagonal().sum()
        # to here

        wpredy = weights.lag_spatial(self.w, self.predy_e)
        wpyTwpy = spdot(wpredy.T, wpredy)
        xTwpy = spdot(self.X.T, wpredy)

        # order of variables is beta, rho, sigma2

        v1 = np.vstack(
            (xtx / self.sig2, xTwpy.T / self.sig2, np.zeros((1, self.k))))
        v2 = np.vstack(
            (xTwpy / self.sig2, tr2 + tr3 + wpyTwpy / self.sig2, tr1 / self.sig2))
        v3 = np.vstack(
            (np.zeros((self.k, 1)), tr1 / self.sig2, self.n / (2.0 * self.sig2 ** 2)))

        v = np.hstack((v1, v2, v3))

        self.vm1 = la.inv(v)  # vm1 includes variance for sigma2
        self.vm = self.vm1[:-1, :-1]  # vm is for coefficients only

    def _fit_gm(self, yend=None, q=None):
        yend2, q2 = set_endog(self.y, self.X[:, 1:], self.w,
                              yend, q, self.w_lags, self.lag_q)  # assumes constant in first column

        self.kstar = yend2.shape[1]
        # including exogenous and endogenous variables
        z = sphstack(self.X, yend2)
        h = sphstack(self.X, q2)
        # k = number of exogenous variables and endogenous variables
        hth = spdot(h.T, h)
        hthi = la.inv(hth)
        zth = spdot(z.T, h)
        hty = spdot(h.T, self.y)

        factor_1 = np.dot(zth, hthi)
        factor_2 = np.dot(factor_1, zth.T)
        # this one needs to be in cache to be used in AK
        varb = la.inv(factor_2)
        factor_3 = np.dot(varb, factor_1)
        params = np.dot(factor_3, hty)
        self.params = params
        self.varb = varb
        self.zthhthi = factor_1

        self.rho = self.betas[-1]
        self.predy_e, self.e_pred, warn = sp_att(self.w, self.y, self.predy,
                                                 self.yend[:, -1].reshape(self.n, 1), self.rho)
        set_warn(self, warn)

    def summary(self, spat_diag=False, gwk=None, sig2n_k=False):
        if self.ml:
            self.k += 1  # add one b/c we added a parameter
            self.aic = DIAG.akaike(reg=self)
            self.schwarz = DIAG.schwarz(reg=self)
            SUMMARY.ML_Lag(reg=self, w=self.w, vm=self.vm, spat_diag=False)
        else:
            if self.robust:
                self.vm = ROBUST.robust_vm(reg=self, gwk=gwk, sig2n_k=sig2n_k)

            if self.sig2n_k:
                self.sig2 = self.sig2n_k
            else:
                self.sig2 = self.sig2n
            SUMMARY.GM_Lag(reg=self, w=self.w, vm=self.vm, spat_diag=spat_diag)
