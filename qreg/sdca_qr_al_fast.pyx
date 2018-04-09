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
from libc.math cimport fabs
from dataset_fast cimport RowDataset

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
                  double mu,
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
            # True update
            dot += coefs[i*n_dim+j] * datain[ii] * dataout[jj] if j != l \
            else coefs[i*n_dim+j] * (datain[ii] * dataout[jj] + mu)  # ALREADY DONE IN _PRED !!!
            # Update as if mu=0 (without augmentation)
#            dot += coefs[i*n_dim+j] * datain[ii] * dataout[jj]
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


cdef double _pred(double* datain,
                  int* indicesin,
                  int n_nzin,
                  double* dataout,
                  int* indicesout,
                  int n_nzout,
                  int n_dim,
                  int coefi,
                  double mu,
                  double* coefs) nogil:

    cdef int i, j, ii, jj, l
    cdef double dot = 0
    l = coefi - (coefi / n_dim) * n_dim

    for ii in xrange(n_nzin):
        i = indicesin[ii]
        for jj in xrange(n_nzout):
            j = indicesout[jj]
            dot += coefs[i*n_dim+j] * datain[ii] * dataout[jj] if j != l \
            else coefs[i*n_dim+j] * (datain[ii] * dataout[jj] + mu)

    return dot


cdef void _solve_subproblem(double* datain,
                            int* indicesin,
                            int n_nzin,
                            double* dataout,
                            int* indicesout,
                            int n_nzout,
                            double y,
                            double* dcoef,
                            int dcoefi,
                            double* xdm,  # 1.T * dcoef
                            int n_samples,
                            int n_dim,
                            double sqnorm,
                            double scale,
                            double stepsize_factor,
                            double prob,
                            double intercept,
                            double mu,
                            double* primal,
                            double* dual,
                            double* regul):

    cdef double pred, dcoef_old, residual, error, loss, update
    cdef double inv_d_stepsize, mult_stepsize

    dcoef_old = dcoef[dcoefi]

    mult_stepsize = sqnorm * stepsize_factor  # is it the best?
    inv_d_stepsize = (sqnorm + mult_stepsize) / 0.95

    pred = _pred(datain, indicesin, n_nzin,
                 dataout, indicesout, n_nzout,
                 n_dim, dcoefi, mu, dcoef)
    
    residual = y - intercept - pred
#    loss = prob*residual if residual > 0 else (prob-1.)*residual
    update = dcoef_old + residual / inv_d_stepsize
    update = min(scale*prob, update)
    update = max(scale*(prob-1.), update)
    update -= dcoef_old
    dual[0] += (y-intercept) * update  # True dual
#    dual[0] += y * update  # Dual as if intercept=0 (without augmentation)

    # Use accumulated loss rather than true primal objective value, which is
    # expensive to compute.
#    primal[0] += loss * scale

    if update != 0:
        dcoef[dcoefi] += update
        _add_l2(datain, indicesin, n_nzin,
                dataout, indicesout, n_nzout,
                update, n_dim, dcoef, dcoefi, mu, regul)
        xdm[0] += update


#SUPPRIMER INDICESIN, INDICESOUT
def _prox_sdca_al_fit(self,
                   RowDataset Kin,
                   RowDataset Kout,
                   np.ndarray[double, ndim=1] y,
                   np.ndarray[double, ndim=1] dual_coef,
                   double alpha2,
                   double C,
                   double stepsize_factor,
                   np.ndarray[double, ndim=1] probs,
                   np.ndarray[double, ndim=1] intercept,  # Dual vector of the linear constraint: + intercept.T * LC
                   double mu,  # Coef of the L2 penalization of the linear constraint: + mu/2 * ||LC||**2
                   int max_iter,
                   double tol,
                   callback,
                   int n_calls,
                   int verbose,
                   rng):

    cdef int n_samples = Kin.get_n_samples()
    cdef int n_dim = Kout.get_n_features()

    # Variables
    cdef double sigma, scale, primal, dual, regul, gap
    cdef int it, ii, i, j
    cdef int has_callback = callback is not None
    cdef LONG t

    # Pre-compute square norms.
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples*n_dim, dtype=np.float64)
    _sqnorms(Kin, Kout, sqnorms)
    sqnorms += mu

    # Pointers
    cdef double* dcoef = <double*>dual_coef.data
    
    cdef np.ndarray[double, ndim=1] xdm_data
    xdm_data = np.zeros(n_dim, dtype=np.float64)  # 1.T * dcoef
    cdef double* xdm = <double*>xdm_data.data
    for j in xrange(n_dim):
        dot = 0.
        for i in xrange(n_samples):
            dot += dcoef[i*n_dim+j]
        xdm[j] = dot

    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples*n_dim, dtype=np.int32)

    # Data pointers.
    cdef int* indicesin
    cdef double* datain
    cdef int n_nzin
    cdef int* indicesout
    cdef double* dataout
    cdef int n_nzout

    scale = C * 1. / alpha2

    dual = 0
    regul = 0
    prev_obj = np.inf

    t = 0        
    for it in xrange(max_iter):
        primal = 0

        rng.shuffle(sindices)

        for ii in xrange(n_samples*n_dim):
            ij = sindices[ii]
            i = ij / n_dim
            j = ij - i*n_dim

            if sqnorms[i*n_dim + j] == 0:
                continue

            # Retrieve rows
            Kin.get_row_ptr(i, &indicesin, &datain, &n_nzin)
            Kout.get_row_ptr(j, &indicesout, &dataout, &n_nzout)

            _solve_subproblem(datain, indicesin, n_nzin,
                              dataout, indicesout, n_nzout,
                              y[i], dcoef, i*n_dim + j,
                              xdm + j, n_samples, n_dim,
                              sqnorms[i*n_dim + j], scale, stepsize_factor,
                              probs[j], intercept[j], mu,
                              &primal, &dual, &regul)

            if has_callback and t % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    break

            t += 1

#                if has_callback and t % n_calls == 0:
#                    ret = callback(self)
#                    if ret is not None:
#                        break

        # tol is the objective value to reach
#        if tol < 0. and np.mod(it, 1e3) == 0:
#            # Minus dual objective value
#            obj = alpha2 * (regul/2. - dual)
#
#            if verbose:
#                print "%8d: %5.2e %5.2e" % (it + 1, obj, obj-tol) 
#
#            # Objective value reached
#            if obj <= tol:
#                if verbose:
#                    print "Ground truth objective value reached."
#                break

        if np.mod(it, 1e3) == 0:
            # Minus dual objective value
            obj = alpha2 * (regul/2. - dual)
            dobj = prev_obj - obj
            prev_obj = obj

            if verbose:
                print "%8d: %5.2e %5.2e" % (it + 1, obj, dobj / n_samples) 
            
            if np.abs(dobj) / n_samples <= tol:
                break
    else:
        if verbose:
            print "Stop before convergence."
