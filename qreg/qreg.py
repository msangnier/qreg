# encoding: utf-8
# Author: Maxime Sangnier
# License: BSD
import sys 

import numpy as np
import scipy.spatial.distance as dist
from scipy.linalg import eigvalsh
from scipy.stats import norm
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from cvxopt import matrix, solvers
from .dataset_fast import get_dataset
from .sdca_qr_fast import _prox_sdca_intercept_fit
from .sdca_qr_al_fast import _prox_sdca_al_fit

import time
import warnings

# time.clock() has been removed in Python 3.8+
# See: https://docs.python.org/3/whatsnew/3.8.html#api-and-feature-removals
if sys.version_info >= (3,8):
    get_time = time.perf_counter
else:
    get_time = time.clock


def toy_data(n=50, t_min=0., t_max=1.5, noise=1., probs=[0.5]):
    """
    Parameters
    n: number of points (t, y)
    t_min: minimum input data t
    t_max: maximum input data t
    noise: noise level
    probs: probabilities (quantiles levels)

    Returns:
    x: sorted random data in [t_min, t_max]
    y: targets corresponding to x (following a noisy sin curve)
    q: true quantiles corresponding to x
    """
    t_down, t_up = 0., 1.5  # Bounds for the noise
    t = np.random.rand(n) * (t_max-t_min) + t_min
    t = np.sort(t)
    pattern = -np.sin(2*np.pi*t)  # Pattern of the signal
    enveloppe = 1 + np.sin(2*np.pi*t/3)  # Enveloppe of the signal
    pattern = pattern * enveloppe
    # Noise decreasing std (from noise+0.2 to 0.2)
    noise_std = 0.2 + noise*(t_up - t) / (t_up - t_down)
    # Gaussian noise with decreasing std
    add_noise = noise_std * np.random.randn(n)
    observations = pattern + add_noise
    quantiles = [pattern + norm.ppf(p, loc=np.zeros(n),
                                    scale=np.fabs(noise_std)) for p in probs]
    return t, observations, quantiles


def proj_dual(coefs, C, probs):
    n = coefs.shape[1]
    for it in range(100):
        # Project onto the hyperplan
        coefs = np.asarray([x - x.sum() / n for x in coefs])
        coefs = np.asarray([np.fmin(C*probs, np.fmax(C*(probs-1), x)) for x in coefs.T]).T

    return coefs


class QRegressor(BaseEstimator):
    def __init__(self, C=1, probs=[0.5], eps=0., kernel='rbf', gamma_in=None,
                 gamma_out=0., alg='coneqp', max_iter=100, tol=1e-6, lag_tol=1e-4,
                 stepsize_factor=10., callback=None,
                 n_calls=None, verbose=False, random_state=None,
                 coefs_init="svr", nc_const=False, al_max_time=180.,
                 max_time=None, n_gap=None, gap_time_ratio=1e-3,
                 active_set=True, sv_tol=1e-3):
        """
        Quantile Regression.

        C: cost parameter (upper bound of dual variables). Positive scalar.
        probs: probabilities (quantiles levels)
        eps: threshold for the epsilon-loss (if used)
        kernel: input kernel ('rbf' or 'linear')
        gamma_in: gamma parameter for the input RBF kernel
        gamma_out: gamma parameter for the output RBF kernel
        alg: algorithm, which can be:
            - 'qp': CVXOPT (alternate optimization when eps > 0)
            - 'coneqp': CVXOPT (cone programming when eps > 0)
            - 'sdca': "A Coordinate Descent Primal-Dual Algorithm with Large
            Step Size and Possibly Non Separable Functions", by Olivier Fercoq
            and Pascal Bianchi
            - 'al': augmented Lagrangian
            - 'mtl': multi-task learning ("Parametric Task Learning", by Ichiro
            Takeuchi, Tatsuya Hongo, Masashi Sugiyama and Shinichi Nakajima)
        max_iter: maximum number of iterations
        tol: prescribed tolerance
        lag_tol: prescribed for the outer loop of the augmented Lagrangian
            algorithm and QP eps
        stepsize_factor
        callback
        n_calls
        verbose
        random_state
        coefs_init: initial dual coefficients (numpy array, n_probs, n_samples))
            If None, initialize with 0. If "svr", initialize with esp-conditional
            median (scikit-learn SVR)
        nc_const: add non-crossing consraints when set to true (only available
            with alg='qp')
        al_max_time: maximum training time (seconds) for al algorithm
        max_time: maximum training time (seconds) for sdca algorithm
        n_gap: number of iterations between two dual gap check (if None, automatic)
        gap_time_ratio: ratio time to compute dual gap / time for n_gap iterations
        (this quantity is used to adjust automatically n_gap)
        active_set: whether to use active set or not
        sv_tol: tolerance for detecting support vector before prediction
        """
        self.C = C
        self.probs = probs
        self.eps = eps
        self.kernel = kernel
        self.gamma_in = gamma_in
        self.gamma_out = gamma_out
        self.alg = alg
        self.alpha = 1.0  # Do not change
        self.max_iter = max_iter
        self.tol = tol
        self.lag_tol = lag_tol
        self.stepsize_factor = stepsize_factor
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.random_state = random_state
        self.coefs_init = coefs_init
        self.nc_const = nc_const
        self.al_max_time = al_max_time
        self.max_time = max_time
        self.n_gap = n_gap
        self.gap_time_ratio = gap_time_ratio
        self.status = ""  # Resolution status
        self.active_set = active_set
        self.sv_tol = sv_tol

    def predict(self, X):
        """
        Predict the conditional quantiles

        Parameters:
        X: data in rows (numpy array)

        Returns:
        y: prediction for each prescribed quantile levels
        """

        X = np.asarray(X)
        if X.ndim == 1:
#            X = np.asarray([X]).T
            # Data has a single feature
            X = X.reshape(-1, 1)

        # Indexes of support vectors
        ind_sv = self.ind_sv()

        # Compute kernels
        if self.kernel == 'rbf':
            Din = dist.cdist(self.X[ind_sv, :], X, 'sqeuclidean')
            Kin = np.exp(-self.gamma_in * Din)
        else:  # Linear kernel
            Kin = np.dot(self.X[ind_sv, :], self.D.dot(X.T))

        Dout = dist.pdist(np.asarray([self.probs]).T, 'sqeuclidean')
        Kout = np.exp(-self.gamma_out * dist.squareform(Dout)) \
            if self.gamma_out != np.inf else np.eye(np.size(self.probs))

        pred = np.dot(np.dot(Kout, self.coefs[:, ind_sv]), Kin).T
        pred += self.intercept
        return pred.T

    def fit(self, X, y):
        """
        Fit the model.

        X: data in rows (numpy array)
        y: targets in rows (numpy array)
        """

        # Was in __init__ before
        self.kernel = self.kernel.lower()
        self.probs = np.asarray(self.probs)
        self.max_iter = int(self.max_iter)
        if self.max_time is None:
            self.max_time = 0
        if self.n_gap is None:
            self.n_gap = 0
        if self.nc_const and self.alg != 'qp':
            self.alg = 'qp'
            warnings.warn("alg set to 'qp' (this is the only available " + \
                "algorithm to deal with the non-crossing consraints)")

        if self.kernel != 'rbf' and self.kernel != 'linear':
            raise ValueError('Choose kernel between rbf and linear.')

        if self.alg == 'mtl':
            self.kernel = 'linear'
            self.gamma_out = np.inf
            self.gamma_in = None

        # Data refactoring
        self.X = np.asarray(X)
        if self.X.ndim == 1:
#            self.X = np.asarray([X]).T
            # Data has a single feature
            self.X = self.X.reshape(-1, 1)
        y = np.ravel(y)

        # If no gamma_in specified, take 0.5 / q, where q is the 0.7-quantile
        # of the squared distances
        if self.kernel == 'rbf':
            Din = dist.pdist(self.X, 'sqeuclidean')
            if self.gamma_in is None:
                self.gamma_in = 1. / (2. * np.percentile(Din, 70.))

        # Compute kernels
        if self.kernel == 'rbf':
            Kin = np.exp(-self.gamma_in * dist.squareform(Din))
        else:  # Linear kernel
            self.D = np.eye(self.X.shape[1])
            Kin = np.dot(self.X, self.D.dot(self.X.T))

        Dout = dist.pdist(np.asarray([self.probs]).T, 'sqeuclidean')
        Kout = np.exp(-self.gamma_out * dist.squareform(Dout)) \
            if self.gamma_out != np.inf else np.eye(np.size(self.probs))

        # Check algorithm
        if self.eps > 0 and self.alg != 'qp' and self.alg != 'coneqp' and self.alg != 'sdca':
            raise ValueError('Use qp or sdca for epsilon quantile regression.')
        if self.nc_const and self.eps > 0:
            raise ValueError('Not implemented yet.')

        # Initialization
        # For QP, it seems to slow down convergence.
        if self.coefs_init is None:
            coefs_init = None
        elif isinstance(self.coefs_init, str) and self.coefs_init.lower() == "svr":
           # Estimate condition median
           svr = SVR(C=self.C/2, kernel="precomputed", epsilon=self.eps)
           svr.fit(Kin, y)
           svr_dual = np.zeros(y.shape)
           svr_dual[svr.support_] = svr.dual_coef_[0, :]
           coefs_init = np.kron(svr_dual, np.ones(np.size(self.probs)))
        else:
           coefs_init = self.coefs_init.T.ravel()

        # Choose the algorithm
        if self.alg == 'qp':  # Off-the-shelf solver (cvxopt)
            if self.nc_const:
                self.qp_nc2(Kin, Kout, y)
            else:
                K = np.kron(Kin, Kout)
                self.qp_eps(K, y) #, coefs_init)
        elif self.alg == 'coneqp':
            if self.nc_const:
                self.qp_nc2(Kin, Kout, y)
            else:
                K = np.kron(Kin, Kout)
                self.coneqp_eps(K, y)
        elif self.alg == 'sdca':  # Stochastic dual coordinate descent
            self.sdca(Kin, Kout, y, coefs_init)
        elif self.alg == 'al':
            self.al(Kin, Kout, y, 1, coefs_init)
        elif self.alg == 'penal':
            self.al(Kin, Kout, y, 4, coefs_init)
        elif self.alg == 'mtl':
            self.mtl(y)
            # Recompute the kernel with learned D
            Kin = np.dot(self.X, self.D.dot(self.X.T))
        else:
            raise ValueError('Unknown algorithm')

        # When there is no additional constraints, the quantile property is
        # satisfied.
        if not self.nc_const:
            # Make the dual point feasible (Mainly for SDCA)
            self.coefs = proj_dual(self.coefs, self.C, self.probs)

            # Set the intercept
            # Erase the previous intercept before prediction
            self.intercept = 0.
            # For usual quantile prediction
            if self.eps == 0.:
                self.intercept = [
                    np.percentile(y-pred, 100.*prob) for
                    (pred, prob) in zip(self.predict(self.X), self.probs)]
                self.intercept = np.asarray(self.intercept)
            else:
                # For eps-quantile prediction
                # Use optimality conditions to find:
                # residues = eps * coef / coef_norm.
                # True for coefs that:
                #   - are not 0
                #   - are not on the boundaries
                tol = 1e-3  # Tolerance for boundaries
                group_norm = np.linalg.norm(self.coefs, axis=0)  # Norm of coefs vectors
                ind_supp = np.where(
                        group_norm / (self.probs.size * self.C) > tol
                        )[0]  # Support vectors
                ind_up = np.where(np.all(
                        (self.probs*self.C-self.coefs.T) / self.C > tol,
                        axis=1))[0]  # Not on boundary sup
                ind_down = np.where(np.all(
                        (self.coefs.T - (self.probs-1)*self.C) / self.C > tol,
                        axis=1))[0]  # Not on boundary inf
                # All conditions together: coefs of interest
                ind = list(set(ind_up) & set(ind_down) & set(ind_supp))
                if ind:
                    # Residues without intercept
                    res = y[ind] - self.predict(self.X)[:, ind]
                    # Expected values from dual coefs
                    res_dual = self.eps * self.coefs[:, ind]/group_norm[ind]
                    # Intercept
                    self.intercept = (res-res_dual).mean(axis=1)
#                    print("qreg residual")
#                    print(res)
#                    print("qreg coefs")
#                    print(self.coefs[:, ind])
#                    print("qreg group_norm")
#                    print(group_norm[ind])
                else:
                    # If ind empty, do similarly as quantile regression
                    self.intercept = [
                        np.percentile(y-pred, 100.*prob) for
                        (pred, prob) in zip(self.predict(self.X), self.probs)]
                    self.intercept = np.asarray(self.intercept)

        # Set optimal objective value
        self.obj = 0.5 * np.trace(np.dot(
            self.coefs.T, np.dot(Kout, np.dot(self.coefs, Kin)))) \
            - np.sum(self.coefs * y)
        self.obj += self.eps * np.linalg.norm(self.coefs, axis=0).sum()

    def score(self, X, y, sample_weight=None):
        # Pinball loss
        return 1 - self.pinball_loss(self.predict(X), y).mean()
        # Pinball loss + Indicator (crossing_loss)
#        p = self.predict(X)
#        return 1 - self.pinball_loss(p, y).mean() + \
#            100. * self.crossing_loss(p).sum()

    def qp_nc(self, Kin, Kout, y):
        ind = np.argsort(self.probs)  # Needed to sort constraints on quantile levels

        K = np.kron(Kin, Kout)
        p = np.size(self.probs)  # Number of quantiles to predict
        n = K.shape[0]  # Number of coefficients
        m = n / p  # Number of training instances
        probs = np.kron(np.ones(m), self.probs)  # Quantiles levels

        D = -np.eye(p) + np.diag(np.ones(p-1), 1)  # Difference matrix
        D = np.delete(D, -1, 0)
        D = D.T[np.argsort(ind)].T

        U = np.kron(Kin, np.dot(Kout, D.T))  # Uper and lower part
        L = np.kron(Kin, np.dot(D, np.dot(Kout, D.T)))  # Right-lower part

        K = matrix(np.r_[np.c_[K, U], np.c_[U.T, L]])  # Quad. part of the obj.
        q = matrix(np.r_[-np.kron(y, np.ones(p)), np.zeros(m*(p-1))])  # Linear part of the objective
        G = matrix(np.r_[np.c_[np.eye(n), np.zeros((n, m*(p-1)))],
                               np.c_[-np.eye(n), np.zeros((n, m*(p-1)))],
                               np.c_[np.zeros((m*(p-1), n)), -np.eye(m*(p-1))]])  # LHS of the inequ. constr.
        h = matrix(np.r_[self.C*probs, self.C*(1-probs), np.zeros(m*(p-1))])  # RHS of the inequ.
        A = matrix(np.c_[np.kron(np.ones(m), np.eye(p)),
                         np.kron(np.ones(m), D.T)])  # LHS of the equ. constr.
        b = matrix(np.zeros(p))  # RHS of the equality constraint

#        The following parameters control the execution of the default solver.
#        options['show_progress'] True/False (default: True)
#        options['maxiters'] positive integer (default: 100)
#        options['refinement']  positive integer (default: 0)
#        options['abstol'] scalar (default: 1e-7)
#        options['reltol'] scalar (default: 1e-6)
#        options['feastol'] scalar (default: 1e-7)
#        Returns:
#        {'dual infeasibility'
#         'dual objective'
#         'dual slack'
#         'gap'
#         'iterations'
#         'primal infeasibility'
#         'primal objective'
#         'primal slack'
#         'relgap'
#         's': <0x1 matrix, tc='d'>,
#         'status'
#         'x'
#         'y'
#         'z'
        solvers.options['show_progress'] = self.verbose
        if self.tol > 0:
            solvers.options['reltol'] = self.tol
        self.time = get_time()  # Store beginning time
        sol = solvers.qp(K, q, G, h, A, b)  # Solve the dual opt. problem
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = np.reshape(sol['x'][:n], (m, p)).T
        self.coefs += np.dot(D.T, np.reshape(sol['x'][n:], (m, p-1)).T)
        self.sol = sol

        # Set the intercept (the quantile property is not verified)
        self.intercept = np.asarray(sol['y']).squeeze()

    def qp_nc2(self, Kin, Kout, y):
        ind = np.argsort(self.probs)  # Needed to sort constraints on quantile levels

        K = np.kron(Kin, Kout)
        p = np.size(self.probs)  # Number of quantiles to predict
        n = K.shape[0]  # Number of coefficients
        m = n / p  # Number of training instances
        l = m * (p-1)  # Number of non-crossing dual variables
        probs = np.kron(np.ones(m), self.probs)  # Quantiles levels

        D = -np.eye(p) + np.diag(np.ones(p-1), 1)  # Difference matrix
        D = np.delete(D, -1, 0)
        D = D.T[np.argsort(ind)].T

        K = matrix(np.r_[np.c_[K, np.zeros((n, l))], np.zeros((l, n+l))])  # Quad. part of the obj.
        q = matrix(np.r_[-np.kron(y, np.ones(p)), np.zeros(l)])  # Linear part of the objective
        G = matrix(np.r_[np.c_[np.eye(n), -np.kron(np.eye(m), D.T)],
                         np.c_[-np.eye(n), np.kron(np.eye(m), D.T)],
                         np.c_[np.zeros((l, n)), -np.eye(l)]])  # LHS of the inequ. constr.
        h = matrix(np.r_[self.C*probs, self.C*(1-probs), np.zeros(m*(p-1))])  # RHS of the inequ.
        A = matrix(np.c_[np.kron(np.ones(m), np.eye(p)), np.zeros((p, l))])  # LHS of the equ. constr.
        b = matrix(np.zeros(p))  # RHS of the equality constraint

        # See qp_nc for usage instruction
        solvers.options['show_progress'] = self.verbose
        if self.tol > 0:
            solvers.options['reltol'] = self.tol
        self.time = get_time()  # Store beginning time
        sol = solvers.qp(K, q, G, h, A, b)  # Solve the dual opt. problem
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = np.reshape(sol['x'][:n], (m, p)).T
        self.sol = sol

        # Set the intercept (the quantile property is not verified)
        self.intercept = np.asarray(sol['y']).squeeze()

    def qp(self, K, y):
        p = np.size(self.probs)  # Number of quantiles to predict
        n = K.shape[0]  # Number of variables
        probs = np.kron(np.ones(n//p), self.probs)  # Quantiles levels

        K = matrix(K)  # Quadratic part of the objective
        q = matrix(-np.kron(y, np.ones(p)))  # Linear part of the objective
        G = matrix(np.r_[np.eye(n), -np.eye(n)])  # LHS of the inequ. constr.
        h = matrix(np.r_[self.C*probs, self.C*(1-probs)])  # RHS of the inequ.
        A = matrix(np.kron(np.ones(n//p), np.eye(p)))  # LHS of the equ. constr.
        b = matrix(np.zeros(p))  # RHS of the equality constraint

        # See qp_nc for usage instruction
        solvers.options['show_progress'] = self.verbose
        if self.tol > 0:
            solvers.options['reltol'] = self.tol
#            solvers.options['feastol'] = self.tol * 1./10
        self.time = get_time()  # Store beginning time
        sol = solvers.qp(K, q, G, h, A, b)  # Solve the dual opt. problem
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = np.reshape(sol['x'], (n//p, p)).T
        self.sol = sol

        # Set the intercept
#        self.intercept = np.asarray(sol['y']).squeeze()

        # Set optimal objective value
        # Either this
#        self.obj = np.asarray(0.5 * sol['x'].T * K * sol['x'] \
#            + q.T * sol['x'])
#        self.obj = float(self.obj.squeeze())
        # Or that
#        self.obj = sol['primal objective']

    def qp_eps(self, K, y): #, coefs_init):
        p = np.size(self.probs)  # Number of quantiles to predict
        n = K.shape[0]  # Number of variables
        probs = np.kron(np.ones(n//p), self.probs)  # Quantiles levels

        q = matrix(-np.kron(y, np.ones(p)))  # Linear part of the objective
        G = matrix(np.r_[np.eye(n), -np.eye(n)])  # LHS of the inequ. constr.
        h = matrix(np.r_[self.C*probs, self.C*(1-probs)])  # RHS of the inequ.
        A = matrix(np.kron(np.ones(n//p), np.eye(p)))  # LHS of the equ. constr.
        b = matrix(np.zeros(p))  # RHS of the equality constraint
        # Initialization is disabled because it seems to slow down convergence
#        initvals = None if self.coefs_init is None else matrix(coefs_init)
        initvals = None

        # See qp_nc for usage instruction
        solvers.options['show_progress'] = self.verbose
        if self.tol > 0:
            solvers.options['reltol'] = self.tol
#            solvers.options['feastol'] = self.tol * 1./10

        self.time = get_time()  # Store beginning time
        if self.eps == 0:
            K = matrix(K)  # Quadratic part of the objective
            sol = solvers.qp(K, q, G, h, A, b, initvals=initvals)  # Solve the dual opt. problem
            coefs = np.reshape(sol['x'], (n//p, p)).T
        else:
            solvers.options['show_progress'] = False
            mu = np.ones(n//p)  # Penalty for l1-l2 norm
            coefs = np.r_[0]  # Initialization for computing improvement

            start_it = time.process_time()
            for it in range(self.max_iter):
                mu = self.eps / mu
                Kmu = matrix(K + np.diag(np.kron(mu, np.ones(p))))  # Quadratic part of the objective
                sol = solvers.qp(Kmu, q, G, h, A, b, initvals=initvals)  # Solve the dual opt. problem
                improvement = np.linalg.norm(coefs.T.ravel() -
                                             np.asarray(sol['x']).ravel()) / (self.C*p)
                coefs = np.reshape(sol['x'], (n//p, p)).T
                if self.verbose:
                    print("it: %d   improvement: %0.2e" % (it, improvement))
                if improvement < self.lag_tol:
                    break
                if self.max_time > 0 and time.process_time() - start_it >self.max_time:
                    break
                # Warm-start is disabled because it seems to slow down convergence
#                initvals = sol['x']
                mu = np.linalg.norm(coefs, axis=0)
                mu[mu < 1e-32] = 1e-32
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = coefs
        self.sol = sol

    def coneqp_eps(self, K, y): #, coefs_init):
        p = np.size(self.probs)  # Number of quantiles to predict
        n = K.shape[0]  # Number of variables
        m = n//p  # Number of points
        probs = np.kron(np.ones(m), self.probs)  # Quantiles levels

        # Initialization is disabled because it seems to slow down convergence
#        initvals = None if self.coefs_init is None else matrix(coefs_init)
        initvals = None

        # See qp_nc for usage instruction
        solvers.options['show_progress'] = self.verbose
        solvers.options['maxiters'] = self.max_iter
        if self.tol > 0:
            solvers.options['reltol'] = self.tol
#            solvers.options['feastol'] = self.tol * 1./10

        self.time = get_time()  # Store beginning time
        if self.eps == 0:
            K = matrix(K)  # Quadratic part of the objective
            q = matrix(-np.kron(y, np.ones(p)))  # Linear part of the objective
            G = matrix(np.r_[np.eye(n), -np.eye(n)])  # LHS of the inequ. constr.
            h = matrix(np.r_[self.C*probs, self.C*(1-probs)])  # RHS of the inequ.
            A = matrix(np.kron(np.ones(m), np.eye(p)))  # LHS of the equ. constr.
            b = matrix(np.zeros(p))  # RHS of the equality constraint

            sol = solvers.qp(K, q, G, h, A, b, initvals=initvals)  # Solve the dual opt. problem
            coefs = np.reshape(sol['x'], (m, p)).T
        else:
            def buildG(m, p):
                n = m*p

                # Get the norm bounds (m last variables)
                A = np.zeros(p+1)
                A[0] = -1
                A = np.kron(np.eye(m), A).T
                # Get the m p-long vectors
                B = np.kron(np.eye(m), np.c_[np.zeros(p), np.eye(p)].T)
                # Box constraint
                C = np.c_[np.r_[np.eye(n), -np.eye(n)], np.zeros((2*n, m))]
                # Set everything together
                C = np.r_[C, np.c_[B, A]]
                return C

            # 2*n non-negative variables
            # [p+1]*m SOC variables

            K = matrix(np.r_[np.c_[K, np.zeros((n, m))], np.zeros((m, n+m))])  # Quadratic part of the objective
            q = matrix(np.r_[-np.kron(y, np.ones(p)), np.ones(m)*self.eps])  # Linear part of the objective
            G = matrix(buildG(m, p))  # LHS of the inequ. constr.
            h = matrix(np.r_[self.C*probs, self.C*(1-probs), np.zeros(m*(p+1))])  # RHS of the inequ.
            A = matrix(np.c_[np.kron(np.ones(m), np.eye(p)), np.zeros((p, m))])  # LHS of the equ. constr.
            b = matrix(np.zeros(p))  # RHS of the equality constraint
            dims = {'l': 2*n, 'q': [p+1]*m, 's': []}

            sol = solvers.coneqp(K, q, G, h, dims, A, b, initvals=initvals)  # Solve the dual opt. problem
            coefs = np.reshape(sol['x'][:n], (m, p)).T
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = coefs
        self.sol = sol

    def sdca(self, Kin, Kout, y, coefs_init):
        n_samples = Kin.shape[0]
        n_dim = Kout.shape[0]

        # For block descent, step size depends on max eigen value of Kout
        # Same as np.linalg.eigvalsh(Kout)[-1]
        Kout_lambda_max = eigvalsh(Kout, eigvals=(n_dim-1,n_dim-1))[0]

        # Data
        dsin = get_dataset(Kin, order="c")
        dsout = get_dataset(Kout, order="c")

        # Initialization
        # Used if done in fit
        self.coefs = np.zeros(n_dim*n_samples, dtype=np.float64) if \
           self.coefs_init is None else coefs_init
        # What is below was relegated to fit
        # if self.coefs_init is None:
        #     self.coefs = np.zeros(n_dim*n_samples, dtype=np.float64)
        # elif isinstance(self.coefs_init, str) and self.coefs_init.lower() == "svr":
        #     # Estimate condition median
        #     svr = SVR(C=self.C/2, kernel="precomputed", epsilon=self.eps)
        #     svr.fit(Kin, y)
        #     svr_dual = np.zeros(y.shape)
        #     svr_dual[svr.support_] = svr.dual_coef_[0, :]
        #     self.coefs = np.kron(svr_dual, np.ones(n_dim))
        # else:
        #     self.coefs = self.coefs_init.T.ravel()

        # Array for objective values
#        inner_obj = np.ones(self.max_iter)

        # Some Parameters
        n_calls = n_samples if self.n_calls is None else self.n_calls
        rng = check_random_state(self.random_state)
        status = np.zeros(1, dtype=np.int16)

        # Call to the solver
        self.time = get_time()  # Store beginning time
        _prox_sdca_intercept_fit(self, dsin, dsout, y, self.coefs, self.alpha,
                                 self.C, self.eps, self.stepsize_factor,
                                 self.probs, self.max_iter, self.tol,
                                 self.callback, n_calls, self.max_time,
                                 self.n_gap, self.gap_time_ratio,
                                 self.verbose, rng, status, self.active_set,
                                 Kout_lambda_max)
#                                , inner_obj)
        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = np.reshape(self.coefs, (n_samples, n_dim)).T

        # Save inner objective values
#        self.inner_obj = inner_obj[inner_obj < 0]

        # Resolution status
        if status[0] == 1:
            self.status = "Optimal solution found"
        elif status[0] == 2:
            self.status = "Maximum iteration reached"
        elif status[0] == 3:
            self.status = "Maximum time reached"
        else:
            self.status = ""

        # Set the intercept
#        self.intercept = 0.  # Erase the previous intercept before prediction
#        self.intercept = [np.percentile(y-pred, 100.*prob) for (pred, prob)\
#        in zip(self.predict(self.X), self.probs)]
#        self.intercept = np.asarray(self.intercept)

        # Set optimal objective value
#        self.obj = 0.5 * np.trace(
#            np.dot(self.coefs.T, np.dot(Kout, np.dot(self.coefs, Kin)))) \
#            - np.sum(self.coefs * y)

    def al(self, Kin, Kout, y, mugrow, coefs_init):
        n_samples = Kin.shape[0]
        n_dim = Kout.shape[0]

        dsin = get_dataset(Kin, order="c")
        dsout = get_dataset(Kout, order="c")

        # Initialization
        # Used if done in fit
        coefs = np.zeros(n_dim*n_samples, dtype=np.float64) if \
            self.coefs_init is None else coefs_init
        # coefs = np.zeros(n_dim*n_samples, dtype=np.float64) \
        #     if self.coefs_init is None else self.coefs_init.T.ravel()
        b = np.zeros(n_dim)  # Intercept

        n_calls = n_samples if self.n_calls is None else self.n_calls
        rng = check_random_state(self.random_state)

        # Parameters of the outer loop
        if mugrow > 1:
            mu = 2  # Factor of the Lagrangian penalization
        elif mugrow == 1:
            mu = 10
        else:
            raise ValueError("mugrow >= 1")
        # mugrow = 4  # Growing factor of the penalization
        prev_err = float('inf')  # Previous error for the outer loop

#        dual_tol = np.sqrt(n) * self.C * self.dual_tol  # Inner loop
        # Loop
        self.time = get_time()  # Store beginning time
        for ito in range(self.max_iter):
            _prox_sdca_al_fit(self, dsin, dsout, y, coefs, self.alpha,
                              self.C, self.stepsize_factor, self.probs, b, mu,
                              self.max_iter, self.tol, self.callback, n_calls,
                              self.verbose, rng)

            # Update the intercept
            # Gradient of the objective wrt the intercept
            der = np.reshape(coefs, (n_samples, n_dim)).sum(axis=0)
            b += mu * der  # Intercept update
            mu *= mugrow  # mu update

            # Stopping criterion
            lag_err = np.sum(der**2)  # Dual error
            if lag_err < self.lag_tol or \
                    np.abs(lag_err/prev_err - 1) < self.lag_tol:
                break
            prev_err = lag_err  # Update the previous Lagrangian error

            # tol is the objective value to reach
            if self.tol < 0:
                # Project coefs on the constraints
                proj_coefs = proj_dual(np.reshape(coefs, (n_samples, n_dim)).T,
                                       self.C, self.probs)

                # Compute the objective value
                obj = 0.5 * np.trace(np.dot(
                    proj_coefs.T, np.dot(Kout, np.dot(proj_coefs, Kin)))) \
                    - np.sum(proj_coefs * y)

                if self.verbose:
                    print("it: %d   obj: %0.2f" % (ito, obj))

                # Stopping criterion
                if obj <= self.tol:
                    if self.verbose:
                        print("Ground truth objective value reached.")
                    break

            # Maximum training time
            current_time = get_time() - self.time  # Current training time
            if current_time > self.al_max_time:
                if self.verbose:
                    print("Maximum training time reached")
                break
        else:
            if self.verbose:
                print('Did not converge after {} iterations.'.format(ito+1))

        self.time = get_time() - self.time  # Store training time

        # Set coefs
        self.coefs = np.reshape(coefs, (n_samples, n_dim)).T

        # Set the intercept
#        self.intercept = b

        # Set optimal objective value
#        self.obj = 0.5 * np.trace(
#            np.dot(self.coefs.T, np.dot(Kout, np.dot(self.coefs, Kin))) ) \
#            - np.sum(self.coefs * y)

    def mtl(self, y):
        d = self.X.shape[1]  # Data dimension
        p = self.probs.shape[0]  # Number of tasks
        verbose = self.verbose
        self.verbose = False

        self.D = np.eye(d) / d  # Initialize D
        err = np.inf

        self.time = get_time()  # Store beginning time
        for it in range(self.max_iter):
            Kin = np.dot(self.X, self.D.dot(self.X.T))  # Compute input kernel
            self.qp(np.kron(Kin, np.eye(p)), y)  # Solve QR problem (fixed D)

            if it < self.max_iter-1:
                # Update D
                B = self.coefs.dot(self.X).dot(self.D)  # Coefficients of the linear predictor

                # Update with eigenvalue decomposition
#                C = B.T.dot(B)
#                e, V = np.linalg.eigh(C)  # Eigen values and vectors
#                e[e<0] = 0
#                self.D = V.dot(np.diag(np.sqrt(e)).dot(V.T))
#                self.D /= np.trace(self.D)

                # Update with singular value decomposition
                _, s, V = np.linalg.svd(B)
                s = np.r_[s, np.zeros(max(0, d-p))]
                D = np.dot(V.T, np.diag(s).dot(V)) / s.sum()
                err = np.linalg.norm(D-self.D)
                self.D = D

            if verbose:
                obj = -0.5 * np.trace(np.dot(
                    self.coefs.T, np.dot(self.coefs, Kin))) \
                    + np.sum(self.coefs * y)
                print(it, obj, err, self.tol)

            if err < self.tol:
                if verbose:
                    print("Did converge.")
                break
        else:
            if verbose:
                print('Did not converge after {} iterations.'.format(it+1))

        self.time = get_time() - self.time  # Store training time
        self.verbose = verbose

    def pinball_loss(self, pred, y):
        y = np.ravel(y)
        residual = y - pred
        loss = np.sum([prob*np.fmax(0, res) for (res, prob) in
            zip(residual, self.probs)], axis=1)
        loss += np.sum([(prob-1)*np.fmin(0, res) for (res, prob) in
            zip(residual, self.probs)], axis=1)
        loss = loss * 1./y.size
        return loss

    def qloss(self, pred, y):
        y = np.ravel(y)
        residual = y - pred
        loss = np.sum([res < 0 for (res, prob) in
            zip(residual, self.probs)], axis=1)
        loss = loss * 1./y.size - self.probs
        return loss

    def crossing_loss(self, pred):
        ind = np.argsort(self.probs)
        loss = np.sum([np.fmax(0, -np.diff(res)) for res in pred[ind].T],
                       axis=0)
        loss = loss * 1./pred.shape[1]
        return loss

    def ind_sv(self):
        group_norm = np.linalg.norm(self.coefs, axis=0) / (self.C * len(self.probs))
        return np.where(group_norm > self.sv_tol)[0]

    def num_sv(self):
        return self.ind_sv().size


class QRegMTL(BaseEstimator):
    def __init__(self, gamma_in=None, Creg=None, location=True,
                 n_landmarks=None, **args):
        """
        Quantile Regression with multi-task learning.

        Ref: Parametric Task Learning, by Ichiro Takeuchi, Tatsuya Hongo,
        Masashi Sugiyama and Shinichi Nakajima (NIPS 2013).

        Methodology:
            First, estimate conditional mean function E[Y|X=x] by least-square
            regression, and compute the residual r_i = y_i − E[Y|X=x_i]. Then,
            apply multi-task learning to (x_i, r_i) and estimate a conditional
            quantile function by Q(x|p) = E[Y|X=x] + h(x|p), where h(.|p) is
            the estimated quantile regression fitted to the residuals.

        location: whether to use a location model (as proposed in the paper)
        gamma_in: gamma parameter for the input RBF map
        n_landmarks: number of landmarks for the input mapping. When None,
            use all training points. When less than 1, consider it as a ratio
            of training points. Else it indicates the number of landmarks.
        Creg: cost parameter for the ridge regression (location model).
            Positive scalar. When it is None, use least-square regression.
        C: cost parameter (upper bound of dual variables). Positive scalar.
        probs: probabilities (quantiles levels)
        max_iter: maximum number of iterations
        tol: prescribed tolerance
        """
        self.gamma_in = gamma_in
        self.location = location
        self.Creg = Creg
        self.n_landmarks = n_landmarks

        if 'alg' in args:
            del args['alg']
        self.reg = QRegressor(alg='mtl', **args)

    def predict(self, X):
        """
        Predict the conditional quantiles

        Parameters:
        X: data in rows (numpy array)

        Returns:
        y: prediction for each prescribed quantile levels
        """

        X = np.asarray(X)
        if X.ndim == 1:
#            X = np.asarray([X]).T
            # Data has a single feature
            X = X.reshape(-1, 1)

        # Map the data with RBF kernel
        Din = dist.cdist(X, self.X, 'sqeuclidean')
        X_map = np.exp(-self.gamma_in * Din)

        # Prediction
        pred = self.reg.predict(X_map)
        if self.location:
            pred += self.lsr.predict(X_map) * self.std_residue + \
                self.mean_residue
#        pred += self.lsr.predict(X_map) if self.location else 0

        return pred

    def fit(self, X, y):
        """
        Fit the model.

        X: data in rows (numpy array)
        y: targets in rows (numpy array)
        """

        self.X = np.asarray(X)  # Training data as landmarks
        if self.X.ndim == 1:
#            self.X = np.asarray([X]).T
            # Data has a single feature
            self.X = self.X.reshape(-1, 1)

        # If no gamma_in specified, take 0.5 / q, where q is the 0.7-quantile
        # of the squared distances
        Din = dist.pdist(self.X, 'sqeuclidean')
        if self.gamma_in is None:
            self.gamma_in = 1. / (2. * np.percentile(Din, 70.))

        # Map the data with RBF kernel
        if not self.n_landmarks:  # landmarks = None  =>  Use all data
            X_map = np.exp(-self.gamma_in * dist.squareform(Din))  # All data as landmarks
        else:
            if self.n_landmarks < 1:  # Ratio
                n_landmarks = int(np.floor(self.n_landmarks * self.X.shape[0]))
            else:
                n_landmarks = self.n_landmarks
            L = self.X[np.random.randint(self.X.shape[0], size=n_landmarks)]  # Random landmarks
            Din = dist.cdist(self.X, L, 'sqeuclidean')
            self.X = L  # Store landmarks
            X_map = np.exp(-self.gamma_in * Din)

        # Lest-squares regression
        if self.location:
            self.lsr = LinearRegression() if not self.Creg \
                else Ridge(alpha=1./self.Creg)
            self.lsr.fit(X_map, y)
            residue = y - self.lsr.predict(X_map)

            self.mean_residue = residue.mean()
            self.std_residue = residue.std()
            residue = (residue - self.mean_residue) / self.std_residue
        else:
            self.lsr = None
            self.mean_residue = None
            self.std_residue = None
            residue = y

        # Fit on training data
        self.reg.fit(X_map, residue)

    def score(self, X, y, sample_weight=None):
        # Pinball loss
        return 1 - self.pinball_loss(self.predict(X), y).mean()

    def get_params(self, deep=True):
        p = super(QRegMTL, self).get_params()
        p.update(self.reg.get_params())
        return p

    def set_params(self, **parameters):
        for parameter in ['gamma_in', 'location', 'Creg', 'reg', 'n_landmarks']:
            if parameter in parameters:
                setattr(self, parameter, parameters[parameter])
                del parameters[parameter]
        self.reg.set_params(**parameters)
        return self

    def pinball_loss(self, pred, y):
        return self.reg.pinball_loss(pred, y)

    def qloss(self, pred, y):
        return self.reg.qloss(pred, y)

    def crossing_loss(self, pred):
        return self.reg.crossing_loss(pred)
