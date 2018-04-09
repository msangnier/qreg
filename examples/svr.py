# encoding: utf-8
# Author: Maxime Sangnier
# License: BSD

"""
Quantile regression with epsilon-insensitive loss (comparison to SVR).
"""

import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from qreg import QRegressor, toy_data
from sklearn.svm import SVR


if __name__ == '__main__':
    probs = [0.5]  # Single quantile regression (match SVR)
    eps = 1e-1  # Threshold for epsilon-loss
    C = 1e2  # Trade-off parameter
    gamma_in = 1  # Gaussian parameter for input data
    max_iter = 1e8  # Large enough
    verbose = False

    # Data
    x_train, y_train, z_train = toy_data(50)
    x_train = x_train[:, np.newaxis]  # Make x 2-dimensional

    # Methods to compare
    methods = [('SVR', SVR(C=C, gamma=gamma_in, epsilon=eps)),
               ('SDCA', QRegressor(C=C*2, probs=probs, gamma_in=gamma_in, eps=eps, coefs_init=None,
                                   max_iter=max_iter, verbose=verbose, max_time=3, alg='sdca')),
               ('QP', QRegressor(C=C*2, probs=probs, gamma_in=gamma_in, eps=eps, coefs_init=None,
                                   max_iter=max_iter, verbose=verbose, max_time=3, alg='qp'))]

    # Objective value
    K = np.exp(-gamma_in * squareform(pdist(x_train, 'sqeuclidean')))  # Kernel matrix
    obj_fun = lambda x: 0.5 * x.dot(K.dot(x)) - y_train.dot(x) + eps*np.linalg.norm(x, ord=1)

    # Figure for dual coefs and residues
    plt.figure(figsize=(15, 8))
    plt.plot([0, y_train.size], [eps] * 2, 'k:', label='+eps')
    plt.plot([0, y_train.size], [-eps] * 2, 'k:', label='-eps')
    # plt.plot([0, y_train.size], [0] * 2, 'k-', label='')

    # Do the job
    for name, reg in methods:
        # Fit the model
        reg.fit(x_train, y_train)

        # Get the dual vector and intercept
        if 'svr' in name.lower():
            dual = np.zeros(y_train.shape)
            dual[reg.support_] = reg.dual_coef_[0, :].copy()
            intercept = reg.intercept_[0]
            pred = reg.predict(x_train)
        else:
            dual = reg.coefs[0, :].copy()
            intercept = reg.intercept[0]
            pred = reg.predict(x_train)[0]

        # Print information
        print(name)
        # Objective value
        if 'sdca' in name.lower():
            print("   objective value: %f (inner value: %f)" % (obj_fun(dual), reg.obj))
        else:
            print("   objective value: %f" % obj_fun(dual))
        # Others
        print("   contraint: 0 = %e" % dual.sum())  # Constraint
        print("   intercept: {}".format(intercept))

        # Plot dual coefs and residues
        plt.plot(dual/C, '-*', label="dual "+name)
        plt.plot(y_train-pred, label="residues "+name)

    # Figure for dual coefs and residues
    plt.grid()
    plt.legend(loc="best")
    plt.show()
