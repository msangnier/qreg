# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Maxime Sangnier and Olivier Fercoq from Mathieu Blondel's sdca
# License: BSD

import numpy as np
cimport numpy as np
ctypedef np.int64_t LONG
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free
from dataset_fast cimport RowDataset
from time import process_time

#np.set_printoptions(precision=4)

cdef void _add_l2(double* datain,
                  int* indicesin,
                  int n_nzin,
                  double* dataout,
                  int* indicesout,
                  int n_nzout,
                  double update,
                  int n_dim,
                  double* coefs,
                  int coefi,
                  double* regul) nogil:

    cdef int i, j, ii, jj, l, m
    cdef double dot
    m = coefi / n_dim
    l = coefi - m * n_dim

    dot = 0.
    for ii in xrange(n_nzin):
        i = indicesin[ii]
        for jj in xrange(n_nzout):
            j = indicesout[jj]
            dot += coefs[i*n_dim+j] * datain[ii] * dataout[jj]  # ALREADY DONE IN _PRED !!!
    regul[0] += update * (2*dot - datain[m] * dataout[l] * update)


cdef _sqnorms(RowDataset Kin, RowDataset Kout,
              np.ndarray[double, ndim=1, mode='c'] sqnorms):

    cdef int n_samples = Kin.get_n_samples()
    cdef int n_dim = Kout.get_n_features()
    cdef int i, j, ii, jj

    # Data pointers.
    cdef double* datain
    cdef double* dataout
    cdef int* indicesin
    cdef int* indicesout
    cdef int n_nzin
    cdef int n_nzout
    cdef double tempin, tempout

    for i in xrange(n_samples):
        tempin = 0.
        Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
        for ii in xrange(n_nzin):  # Look for the ith element in Kin(i, :)
            if indicesin[ii] == i:
                tempin = datain[ii]
                break
        for j in xrange(n_dim):
            tempout = 0.
            Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)
            for jj in xrange(n_nzout):  # Look for the jth element in Kout(j, :)
                if indicesout[jj] == j:
                    tempout = dataout[jj]
                    break
            sqnorms[i*n_dim + j] = tempin * tempout


cdef _sqnormsL(RowDataset Kin, double lambda_max,
              np.ndarray[double, ndim=1, mode='c'] sqnorms):

    cdef int n_samples = Kin.get_n_samples()
    cdef int i, ii

    # Data pointers.
    cdef double* datain
    cdef int* indicesin
    cdef int n_nzin
    cdef double tempin

    for i in xrange(n_samples):
        tempin = 0.
        Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
        for ii in xrange(n_nzin):  # Look for the ith element in Kin(i, :)
            if indicesin[ii] == i:
                tempin = datain[ii]
                break
        sqnorms[i] = tempin * lambda_max


cdef double _pred(double* datain,
                  int* indicesin,
                  int n_nzin,
                  double* dataout,
                  int* indicesout,
                  int n_nzout,
                  int n_dim,
                  double* coefs) nogil:

    cdef int i, j, ii, jj
    cdef double dot = 0

    for ii in xrange(n_nzin):
        i = indicesin[ii]
        for jj in xrange(n_nzout):
            j = indicesout[jj]
            dot += coefs[i*n_dim+j] * datain[ii] * dataout[jj]

    return dot

cdef double norm_square(double* y, int y_size) nogil:
    cdef double norm = 0.
    for it in range(y_size):
        norm += y[it]**2
    return norm

cdef void clip(double mu, double* y, double* probs, double scale, int y_size,
               double* clip_y, double* circ_y) nogil:
    for it in range(y_size):
        clip_y[it] = min(scale*probs[it], max(scale*(probs[it]-1), mu*y[it]))
        circ_y[it] = y[it] if clip_y[it] < scale*probs[it] and clip_y[it] > scale*(probs[it]-1) else 0.

cdef double solve_prox_equ(double mu, double l, double* y, double* probs,
                           double scale, int y_size) nogil:
    cdef double tol = 1.48e-08  # Scipy value
    cdef int max_iter = 50  # Scipy value
    cdef double* v = <double*> malloc(y_size*sizeof(double))  # Truncated vector
    cdef double* u = <double*> malloc(y_size*sizeof(double))  # Zero truncation
#    cdef np.ndarray[double, ndim=1] v_data  # Truncated vector
#    cdef np.ndarray[double, ndim=1] u_data  # Zero truncation
#    v_data = np.zeros(y_size, dtype=np.float64)
#    u_data = np.zeros(y_size, dtype=np.float64)
#    cdef double* v = <double*>v_data.data
#    cdef double* u = <double*>u_data.data

#    print("------------------------------------------------------------------")
#    print("mu init", mu)

    for it in range(max_iter):
        clip(mu, y, probs, scale, y_size, v, u)
        v_norm = sqrt(norm_square(v, y_size))
        phi = 1 + l / v_norm - 1/mu  # Objective
        err = fabs(phi)
        if err < tol:
            break
        diff_phi = 1/mu**2 - l*mu*norm_square(u, y_size) / v_norm**3  # Derivative
        mu -= phi / diff_phi  # Newton update
#        print("it", it, "mu", mu, "phi", phi, "diff_phi", diff_phi,
#              "norm", norm_square(y, y_size))
        # Prevent divergence
#        if mu < 0 or mu > 1:
#            print("Error in mu")
#            return solve_prox_equ_bisect(l, y, probs, scale, y_size)

    free(v)
    free(u)
    return mu

#cdef double solve_prox_equ_bisect(double l, double* y, double* probs,
#                           double scale, int y_size):
#    cdef double tol = 1e-12
#    cdef int max_iter = 100
#    cdef np.ndarray[double, ndim=1] v_data  # Truncated vector
#    cdef np.ndarray[double, ndim=1] u_data  # Zero truncation
#    v_data = np.zeros(y_size, dtype=np.float64)
#    u_data = np.zeros(y_size, dtype=np.float64)
#    cdef double* v = <double*>v_data.data
#    cdef double* u = <double*>u_data.data
#
#    mu1 = 1e-6
#    mu2 = 1
#    # Find a negative point
#    for it in range(max_iter):
#        clip(mu1, y, probs, scale, y_size, v, u)
#        v_norm = np.sqrt(norm_square(v, y_size))
#        phi = 1 + l / v_norm - 1/mu1  # Objective
#        if phi < 0:
#            break
#        mu1 /= 10
##    print(mu1, phi)
#    for it in range(max_iter):
#        mu = (mu1+mu2)/2
#        clip(mu, y, probs, scale, y_size, v, u)
#        v_norm = np.sqrt(norm_square(v, y_size))
#        phi = 1 + l / v_norm - 1/mu  # Objective
#        err = abs(phi)
#        if err < tol:
#            break
#        if phi > 0:
#            mu2 = mu
#        else:
#            mu1 = mu
##        print("it", it, "mu", mu, "phi", phi)
#    return mu

cdef void _solve_subproblem(double* datain,
                            int* indicesin,
                            int n_nzin,
                            RowDataset Kout,
                            double* dataout,
                            int* indicesout,
                            int n_nzout,
                            double y,
                            double* dcoef,
                            int dcoefi,
                            double* multiplier,
                            double* residual,
                            double* xdm,  # 1.T * dcoef
                            double* ydm,  # 1.T * multiplier
                            int n_samples,
                            int n_dim,
                            double sqnorm,
                            double scale,
                            double eps,
                            double* group_norm,
                            double* res_norm,
                            int* coef_on_bound,
                            double* approx_mu,
                            double stepsize_factor,
                            double* probs,
                            int i,
                            double* primal,
                            double* dual,
                            double* regul):

    cdef double pred, error, loss, eps_prox, new_norm, gnorm, res_coef
    cdef double multiplier_old, multiplier_update
    cdef double inv_d_stepsize, mult_stepsize
    cdef double tol_bound

    # Updates of dual coefs
    cdef double* update = <double*> malloc(n_dim*sizeof(double))

    mult_stepsize = sqnorm * stepsize_factor  # is it the best?
    inv_d_stepsize = (sqnorm + mult_stepsize) / 0.95
    eps_prox = eps/inv_d_stepsize
    gnorm = group_norm[0]

    res_norm[0] = 0
    coef_on_bound[0] = 1
    tol_bound = 1e-6

    for j in xrange(n_dim):
        multiplier_old = multiplier[j]

        Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)
        pred = _pred(datain, indicesin, n_nzin,
                     dataout, indicesout, n_nzout,
                     n_dim, dcoef)

        # i-th element of the projection of
        # mutiplier + mult_stepsize * dcoef on [1, ..., 1]
        multiplier_update = (ydm[j] + mult_stepsize * xdm[j]) / n_samples
        multiplier_update -= multiplier_old

        residual[j] = y - pred
        update[j] = (dcoef[dcoefi+j] + (
                residual[j] - (multiplier_old + 2. *  multiplier_update))
                / inv_d_stepsize)
        residual[j] -= multiplier_old  # Minus intercept
        res_norm[0] += residual[j]**2

        # Compute the loss (first way to do, an other one is below)
#    #    loss = probs[j]*residual[0] if residual[0] > 0 else (probs[j]-1.)*residual[0]
#        loss = probs[j]*max(0, residual[j]-eps/n_dim) + (probs[j]-1)*min(0, residual[j]+eps/n_dim)
#        primal[0] += loss  # Accumulated loss

        # Update group norm
        gnorm += update[j]**2 - dcoef[dcoefi+j]**2

        # Update multiplier
        if multiplier_update != 0:
            multiplier[j] += multiplier_update
            ydm[j] += multiplier_update

        # Is coef on bound?
        if probs[j] - dcoef[dcoefi+j]/scale > tol_bound and \
            dcoef[dcoefi+j]/scale - probs[j]+1 > tol_bound:
                coef_on_bound[0] = 0

    res_norm[0] = sqrt(res_norm[0])

    # l1-l2 proximal operator + box constraint
    # Method 1 (full)
    if eps > 0.:
        new_norm = sqrt(gnorm)
        if new_norm > eps_prox:
            mu = solve_prox_equ(1-eps_prox/new_norm, eps_prox, update, probs,
                                scale, n_dim)
            for j in xrange(n_dim):
                update[j] *= mu
                # Box constraint projection
                update[j] = min(scale*probs[j], update[j])
                update[j] = max(scale*(probs[j]-1.), update[j])
                group_norm[0] += update[j]**2 - dcoef[dcoefi+j]**2
        else:
            for j in xrange(n_dim):
                update[j] = 0.
                group_norm[0] -= dcoef[dcoefi+j]**2
    else:
        # Box constraint projection
        for j in xrange(n_dim):
            update[j] = min(scale*probs[j], update[j])
            update[j] = max(scale*(probs[j]-1.), update[j])

    # Coef for computing the loss
    res_coef = 1 - min(eps, res_norm[0])/res_norm[0] if res_norm[0]>0 else 0

    for j in xrange(n_dim):
        # Compute the loss (second way to do, more accurate)
        loss = probs[j]*max(0, residual[j]*res_coef) + (probs[j]-1)*min(0, residual[j]*res_coef)
        primal[0] += loss  # Accumulated loss

        update[j] -= dcoef[dcoefi+j]
        dual[0] += y * update[j]

        if update[j] != 0:
            dcoef[dcoefi+j] += update[j]

            Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)
            _add_l2(datain, indicesin, n_nzin,
                    dataout, indicesout, n_nzout,
                    update[j], n_dim, dcoef, dcoefi+j, regul)
            xdm[j] += update[j]

    free(update)

        # Method 2 (totally approximated)
    #    update *= approx_mu
    #    # Box constraint projection
    #    update = min(scale*prob, update)
    #    update = max(scale*(prob-1.), update)
    #    new_norm = group_norm[0]**2 - dcoef_old**2 + update**2

    #    # Method 3 (partially approximated)
    #    if eps > 0.:
    #        new_norm = group_norm[0] + update**2 - dcoef_old**2
    #        if sqrt(new_norm) > eps:
    ##            if approx_mu[0] > 0.:
    ##                update *= approx_mu[0]
    ##            else:
    #            if approx_mu[0] == 0.:
    #                dcoef[dcoefi] = update
    #                mu = solve_prox_equ(1-eps/sqrt(new_norm), eps, dcoef+i*n_dim,
    #                                    probs, scale, n_dim)
    #                dcoef[dcoefi] = dcoef_old
    ##                print("mu vs approx_mu", mu, approx_mu[0])
    #                approx_mu[0] = mu
    ##                update *= mu
    #            update *= approx_mu[0]
    #        else:
    #            update = 0.
    #    # Box constraint projection
    #    update = min(scale*prob, update)
    #    update = max(scale*(prob-1.), update)
    #    group_norm[0] += update**2 - dcoef_old**2

        # Method 4 (partially approximated, best for now)
    #    if eps > 0.:
    #        if approx_mu[0] == 0.:
    #            eps_prox = eps/inv_d_stepsize
    #            new_norm = sqrt(group_norm[0] + update**2 - dcoef_old**2)
    #            if new_norm > eps_prox:
    #                mu_init = 1-eps_prox/new_norm
    #                if abs(0.5-mu_init) < 0.45:
    #                    dcoef[dcoefi] = update
    #                    mu = solve_prox_equ(mu_init, eps_prox, dcoef+i*n_dim,
    #                                        probs, scale, n_dim)
    #                    dcoef[dcoefi] = dcoef_old
    #                    approx_mu[0] = mu
    #                else:
    #                    approx_mu[0] = mu_init  # mu_init is close to the solution
    #                    #when it is close to 0 or 1
    #            else:
    #                approx_mu[0] = 0.
    #        update *= approx_mu[0]

        # Method 5 (partially approximated)
    #    if eps > 0.:
    #        new_norm = sqrt(group_norm[0] + update**2 - dcoef_old**2)
    #        mu = 1 - eps/new_norm if new_norm > eps else 0.
    #        mu = max(0, 1 - eps/sqrt(new_norm))
    #        if abs(0.5-mu) < 0.45:
    #            print("top")
    #            dcoef[dcoefi] = update
    #            mu = solve_prox_equ(mu, eps, dcoef+i*n_dim, probs, scale,
    #                                n_dim)
    #            dcoef[dcoefi] = dcoef_old
    #        update *= mu

    #    # Box constraint projection
    #    update = min(scale*prob, update)
    #    update = max(scale*(prob-1.), update)
    #    group_norm[0] += update**2 - dcoef_old**2


#SUPPRIMER INDICESIN, INDICESOUT
def _prox_sdca_intercept_fit(self,
                   RowDataset Kin,
                   RowDataset Kout,
                   np.ndarray[double, ndim=1] y,
                   np.ndarray[double, ndim=1] dual_coef,
                   double alpha2,
                   double C,
                   double eps,
                   double stepsize_factor,
                   np.ndarray[double, ndim=1] probs,
                   int max_iter,
                   double tol,
                   callback,
                   int n_calls,
                   float max_time,
                   int n_gap,
                   float gap_time_ratio,
                   int verbose,
                   rng,
                   np.ndarray[short int, ndim=1] status,
                   int active_set,
                   double lambda_max):
#                   np.ndarray[double, ndim=1] inner_obj):

    cdef int n_samples = Kin.get_n_samples()
    cdef int n_dim = Kout.get_n_features()

    # Variables
    cdef double sigma, scale, primal, dual, regul, gap, dual_sparsity, old_gn, constraint
    cdef int it, ii, i, j
    cdef int has_callback = callback is not None
    cdef LONG t
    cdef double tol_bound
    cdef int check_gap, perf_active_set, n_act_coord, n_act_coord_prev

    # Pre-compute square norms.
#    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
#    sqnorms = np.zeros(n_samples*n_dim, dtype=np.float64)
#    _sqnorms(Kin, Kout, sqnorms)

    # Pre-compute Lipschitz constants
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples, dtype=np.float64)
    _sqnormsL(Kin, lambda_max, sqnorms)

    # Pointers
    cdef double* dcoef = <double*>dual_coef.data
    cdef double* cprobs = <double*>probs.data
#    cdef double* iobj = <double*>inner_obj.data
    cdef int* cstatus = <int*>status.data

    cdef np.ndarray[double, ndim=1] multiplier_data
    multiplier_data = np.zeros(n_dim*n_samples, dtype=np.float64)
    cdef double* multiplier = <double*>multiplier_data.data

    cdef np.ndarray[double, ndim=1] residual_data
    residual_data = np.zeros(n_dim*n_samples, dtype=np.float64)
    cdef double* residual = <double*>residual_data.data

    cdef np.ndarray[double, ndim=1] ydm_data
    ydm_data = np.zeros(n_dim, dtype=np.float64)  # 1.T * multiplier
    cdef double* ydm = <double*>ydm_data.data

    cdef np.ndarray[double, ndim=1] xdm_data
    xdm_data = np.zeros(n_dim, dtype=np.float64)  # 1.T * dcoef
    cdef double* xdm = <double*>xdm_data.data
    for j in xrange(n_dim):
        dot = 0.
        for i in xrange(n_samples):
            dot += dcoef[i*n_dim+j]
        xdm[j] = dot

    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples, dtype=np.int32)
    sindices_size = n_samples

    cdef np.ndarray[double, ndim=1] group_norm_data
    group_norm_data = np.zeros(n_samples, dtype=np.float64)  # squared norm for each group
    cdef double* group_norm = <double*>group_norm_data.data
    for i in range(n_samples):
        group_norm[i] = norm_square(dcoef+i*n_dim, n_dim)

    cdef np.ndarray[double, ndim=1] res_norm_data
    res_norm_data = np.zeros(n_samples, dtype=np.float64)  # norm for each residue
    cdef double* res_norm = <double*>res_norm_data.data

    cdef np.ndarray[int, ndim=1] coef_on_bound_data
    coef_on_bound_data = np.zeros(n_samples, dtype=np.int32)  # Is coef on box bound?
    cdef int* coef_on_bound= <int*>coef_on_bound_data.data

    cdef np.ndarray[double, ndim=1] mus_data
    mus_data = np.ones(n_samples, dtype=np.float64)  # 1 for eps=0
    cdef double* mus = <double*>mus_data.data

    # Data pointers.
    cdef int* indicesin
    cdef double* datain
    cdef int n_nzin
    cdef int* indicesout
    cdef double* dataout
    cdef int n_nzout
    n_gap_auto = n_gap==0
    if n_gap_auto:
        n_gap = 100

    swap_active_set = 0

    scale = C * 1. / alpha2

#    dual = (y * dual_coef).sum()
    dual = (y * np.reshape(dual_coef, (n_samples, n_dim)).T).sum()
#    dual = 0.
#    for i in xrange(n_samples):
#        dot = 0.
#        for j in xrange(n_dim):
#            dot += dcoef[i*n_dim+j]
#        dual += y[i]*dot

    dual_sparsity = np.sqrt(group_norm_data).sum()
    regul = 0
    for i in xrange(n_samples):
        Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
        for j in xrange(n_dim):
            Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)
            dot = 0.
            for ii in xrange(n_nzin):
                for jj in xrange(n_nzout):
                    dot += dcoef[indicesin[ii]*n_dim+indicesout[jj]] * datain[ii] * dataout[jj]
            regul += dcoef[i*n_dim+j] * dot
    if verbose:
#        print("regul", regul, "dual", dual, "group_norm", dual_sparsity)
        constraint = np.sum(np.fabs(xdm_data))
        obj = alpha2 * (regul/2. - dual + 100.*constraint + eps * dual_sparsity)
        print("Initial obj:", obj)

    ################ Test solve_prox_equ ##############
#    n = 6
#    cdef np.ndarray[double, ndim=1] b
#    cdef np.ndarray[double, ndim=1] yy
#    b = 0.8*np.ones(n)
#    a = b-1
#    l = 2
#    yy = np.random.randn(n)*5
#
#    mu = 0.5
#    for it in range(50):
#        phi = 1 + l / np.linalg.norm(np.fmin(b, np.fmax(a, mu*yy))) - 1/mu
#        v = np.fmin(b, np.fmax(a, mu*yy))
#        u = yy * (v<b) * (v>a)
#        diff_phi = 1/mu**2 - l*mu*np.linalg.norm(u)**2 / np.linalg.norm(v)**3
#        mu -= phi / diff_phi
#
#    print("mu Newton", mu)
#
#    cdef double* cb = <double*>b.data
#    cdef double* cy = <double*>yy.data
#    mu = solve_prox_equ(0.5, l, cy, cb, 1, n)
#    print("mu Newton", mu)
#    mu = solve_prox_equ_bisect(l, cy, cb, 1, n)
#    print("mu bisect", mu)
    ################ Test solve_prox_equ ##############

#    n_indices_max = 200
#    if sindices_size > n_indices_max:
#        rng.shuffle(sindices)
#        sindices = sindices[:n_indices_max]
#        sindices_size = n_indices_max

    t = 0
    i_check_gap = 0
    time_gap = 0
    n_act_coord = 0
    tol_bound = 1e-6

    start_it = process_time()
    for it in xrange(max_iter):
        primal = 0
        n_act_coord_prev = n_act_coord
        n_act_coord = 0

        check_gap = it+1 - (it+1)//n_gap * n_gap == 0  # np.mod(it+1, n_gap)
#        perf_active_set = it+2 - (it+2)//n_gap * n_gap != 0 # Iteration before checking the gap
        perf_active_set = 1-check_gap

        rng.shuffle(sindices)

        # Set values for mu (leave at 1 for first iteration)
#        if eps > 0. and it>0:
#            for i in range(n_samples):
##                print(group_norm[i], norm_square(dcoef+i*n_dim, n_dim))
#                if np.sqrt(group_norm[i]) > eps:
#                    mus[i] = solve_prox_equ(1-eps/np.sqrt(group_norm[i]), eps,
#                       dcoef+i*n_dim, cprobs, scale, n_dim)
#                else:
#                    mus[i] = 0.
#    #            print("mu", i, mus[i], 1-eps/np.sqrt(group_norm[i]), group_norm[i])

        # Reset mus
#        if eps > 0. and it>0:
#            for i in range(n_samples):
#                mus[i] = 0.

#        for ii in xrange(n_samples*n_dim):
#            ij = sindices[ii]
#            i = ij / n_dim  # Sample index
#            j = ij - i*n_dim  # Dimension index
        for ii in xrange(sindices_size):
            i = sindices[ii]
#            mus[i] = 0.
            old_gn = group_norm[i]
            if sqnorms[i] == 0:
                continue

            if active_set==1:
                if it>100 and (\
                    (res_norm[i] < eps*0.9 and \
                    sqrt(group_norm[i]) / (n_dim*scale) < tol_bound) or \
                     (res_norm[i] > eps*1.1 and coef_on_bound[i] == 1)
                    ) and perf_active_set==1:
                    continue

            n_act_coord += 1

            # Retrieve rows
            Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
#            Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)

            _solve_subproblem(datain, indicesin, n_nzin,
                              Kout, dataout, indicesout, n_nzout,
                              y[i], dcoef, i*n_dim,
                              multiplier + i*n_dim,
                              residual + i*n_dim,
                              xdm, ydm, n_samples, n_dim,
                              sqnorms[i], scale, eps,
                              group_norm + i, res_norm + i,
                              coef_on_bound + i,
                              mus + i, stepsize_factor,
                              cprobs, i,
                              &primal, &dual, &regul)

            if has_callback and t % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    break

            t += 1

            if eps > 0.:
                if group_norm[i] < 0.:
                    group_norm[i] = 0.
                dual_sparsity += sqrt(group_norm[i]) - sqrt(old_gn)

#                if has_callback and t % n_calls == 0:
#                    ret = callback(self)
#                    if ret is not None:
#                        break

        # Debug
        # Compute 0.5 * dcoef.T * kron(Kin, Kout) * dcoef -
        #   kron(y, ones(n_dim)).T * dcoef
        # This should be equal to alpha2 * (regul/2. - dual)
#        if np.mod(it, 1e0) == 0:
#            obj = 0.
#            dot = 0.
#            for i in xrange(n_samples):
#                Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
#                for j in xrange(n_dim):
#                    Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)
#                    dot += dcoef[i*n_dim+j] * y[i]
#    #                print "%f < %f < %f" % (scale*(probs[j]-1), scale*dcoef[i*n_dim+j], scale*probs[j])
#                    for ii in xrange(n_samples):
#                        for jj in xrange(n_dim):
#                            obj += datain[ii] * dataout[jj] * dcoef[i*n_dim+j] * dcoef[ii*n_dim+jj]
#            obj *= 0.5
#            obj -= dot
#            print "It: %d   obj: %f" % (it+1, obj)


        ################## Duality gap, print and active set ##################
#        if it+1 - (it+1)//n_gap * n_gap == 0:  # np.mod(it+1, n_gap)
        if check_gap:
            i_check_gap += 1
            start_gap = process_time()

#            constraint = np.sum(np.fabs(xdm_data))
            constraint = 0
            for j in xrange(n_dim):
                constraint += abs(xdm[j])
#            obj = alpha2 * (regul/2. - dual + 100.*(constraint if constraint > 1e-4 else 0.))  # Minus dual objective value
            obj = alpha2 * (regul/2. - dual + 100.*constraint)  # Minus dual objective value
            obj += alpha2 * eps * dual_sparsity

#            iobj[i_check_gap-1] = obj

            # Compute the intercept (not needed anymore since -multiplier_old has been added to residual)
#            rresidual = np.reshape(residual_data, (n_samples, n_dim)).T
            # By duality
#            intercept = np.reshape(multiplier_data, (n_samples, n_dim)).mean(axis=0)
            # Or…
#            if eps == 0.:
#                # Minimize primal problem
#                intercept = [
#                    np.percentile(res, 100.*prob) for (res, prob) in
#                    zip(rresidual, probs)]
#            else:
#                # Use optimality conditions
#                tol_bound = 1e-3  # Tolerance for boundaries
#                coefs = np.reshape(dual_coef, (n_samples, n_dim)).T
#                ind_supp = np.where(np.sqrt(group_norm_data) / (n_dim * C) > tol_bound)[0]  # Support vectors
#                ind_up = np.where(np.all(
#                        (probs*C-coefs.T) / C > tol_bound, axis=1))[0]  # Not on boundary sup
#                ind_down = np.where(np.all(
#                        (coefs.T - (probs-1)*C) / C > tol_bound, axis=1))[0]  # Not on boundary inf
#                # All conditions together: coefs of interest
#                # Intersection of ind_up, in_down and ind_supp
#                ind = [el for el in ind_up if el in ind_down and el in ind_supp]
##                print("ind in sdca", ind)
#                if ind:
#                    # Residues without intercept - expected values from dual coefs
#                    intercept = (rresidual[:, ind]\
#                                      -eps * coefs[:, ind]/\
#                                      np.sqrt(group_norm_data[ind])).mean(axis=1)
##                    print("sdca residual")
##                    print(rresidual[:, ind])
##                    print("sdca coefs")
##                    print(coefs[:, ind])
##                    print("sdca group_norm")
##                    print(np.sqrt(group_norm_data[ind]))
#                else:
#                    # If ind empty, do similarly as quantile regression
#                    intercept = [
#                        np.percentile(res, 100.*prob) for (res, prob) in
#                        zip(rresidual, probs)]
##            print("intercept")
##            print(intercept)
##            print("multiplier")
##            print(np.reshape(multiplier_data, (n_samples, n_dim)).mean(axis=0))

#            rresidual = (rresidual.T - intercept).T

            # Compute the primal objective (approximated for eps-loss)
#            primal2 = np.sum([
#                prob*np.fmax(0, res-eps) for (res, prob) in
#                zip(rresidual, probs)])
#            primal2 += np.sum([
#                (prob-1)*np.fmin(0, res+eps) for (res, prob) in
#                zip(rresidual, probs)])
#            print(primal2, primal)
            # Use accumulated loss
            primal2 = alpha2 * (regul/2. + primal*scale)
            gap = (primal2 + obj) / (C * n_samples)  # Dual gap
            if gap < 0.:
                gap = 1

            # Active set (if enabled and intercept obtained by optimality conditions)
#            if active_set > 0:
#                if swap_active_set:
#                    tol_bound = 1e-6  # Tolerance for boundaries (redefinition)
##                    coefs = np.reshape(dual_coef, (n_samples, n_dim)).T
##                    rresidual = np.reshape(residual_data, (n_samples, n_dim)).T
##                    rresidual_norm = np.linalg.norm(rresidual, axis=0)  # Residues norm
##                    print("residual norm")
##                    print(rresidual_norm)
##                    print("online residual norm")
##                    print(res_norm_data)
#
#                    # Points with small residues and zero coefs
##                    ind_null = np.where(rresidual_norm_data < eps-tol_bound)[0]
##                    ind_null_coef = np.where(np.sqrt(group_norm_data) / (n_dim * C) < tol_bound)[0]
##                    ind_null_coef = [e for e in ind_null if e in ind_null_coef]
#                    ind_null_coef = [j for j in xrange(n_samples) if
#                                     res_norm[j] < eps*0.9 and
#                                     sqrt(group_norm[j]) / (n_dim*C) < tol_bound]
#
#                    # Points with large residues and coefs on box borders
##                    ind_bound = np.where(res_norm_data > eps*1.1)[0]
##                    ind_bound_coef = np.where(np.all(np.logical_or(
##                            (probs*C-coefs.T) / C < tol_bound,
##                            (coefs.T - (probs-1)*C) / C < tol_bound), axis=1))[0]
##                    ind_bound_coef = [e for e in ind_bound if e in ind_bound_coef]
#                    ind_bound_coef = [j for j in xrange(n_samples) if
#                                     res_norm[j] > eps*1.1 and
#                                     coef_on_bound[j] == 1]
#
#                    # All that points satisfy optimality conditions
#                    ind_all_coef = ind_null_coef + ind_bound_coef
#                    sindices = np.delete(np.arange(n_samples, dtype=np.int32),
#                                         ind_all_coef)
#                    sindices_size = sindices.size
##                    print(np.sort(sindices))
#                else:
#                    if it > 1:
#                        swap_active_set = 1
#
#                    sindices = np.arange(n_samples, dtype=np.int32)
#                    sindices_size = n_samples
#                swap_active_set = 1 - swap_active_set  # 0 <-> 1

#                swap_active_set += 1
#                if swap_active_set > 2:
#                    swap_active_set = 0

#                if sindices_size > n_indices_max:
#                    rng.shuffle(sindices)
#                    sindices = sindices[:n_indices_max]
#                    sindices_size = n_indices_max

    #            print("all coefs", ind_all_coef)
    #            print("active indexes", sindices)
    #            print(coefs[:, ind_all_coef] / C)
    #            print("# active coord", sindices.size)

            # Automatic tuning such that the time of computing the duality gap
            # don't exceed 100*gap_time_ratio % of the total time
            end_gap = process_time()
            elapsed_time = (end_gap - start_it) / i_check_gap  # Time between 2 checks
            time_gap = ((i_check_gap-1)*time_gap + end_gap - start_gap) / i_check_gap
            if n_gap_auto:
                n_gap = max(100, int(n_gap * time_gap / (elapsed_time * gap_time_ratio)))

#            if swap_active_set:
#                n_gap = 10

            if verbose:
                print "%8d: %5.2e (gap) %5.2f (obj) %5.2e (constraint) %5.2f (gap time ratio) %d (# act coord)" % (it + 1, gap, obj, constraint, 100.*time_gap/elapsed_time, n_act_coord_prev)

            # Stopping criterion
            if gap <= tol:
                if verbose:
                    print "Optimal solution found."
                status[0] = 1
                break

        if max_time > 0 and process_time() - start_it > max_time:
            if verbose:
                print "Max time reached."
            status[0] = 3
            break

#        # tol is the objective value to reach
#        if (verbose or tol < 0.) and np.mod(it+1, 1e3) == 0:
#            # Minus dual objective value
#            obj = alpha2 * (regul/2. - dual + 100.*np.sum(np.fabs(xdm_data)))
##            obj = alpha2 * (regul/2. - dual)
#
#            if verbose:
#                print "%8d: %5.2e %5.2e" % (it + 1, obj-tol,
#                                            np.sum(np.fabs(xdm_data)))
#
#            # Objective value reached
#            if tol < 0. and obj <= tol:
#                if verbose:
#                    print "Ground truth objective value reached."
#                break

    else:
        if verbose:
            print "Max iteration reached."
        status[0] = 2
    # Debug sparsity (norm accumulation)
#    print(group_norm_data)
#    coefs = np.reshape(dual_coef, (n_samples, n_dim)).T
#    print(np.sqrt((coefs**2).sum(axis=0)))

#    print("sindices", np.sort(sindices))

