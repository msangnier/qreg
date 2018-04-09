# -*- coding: utf-8 -*-
"""
Example of how to use the Quantile Regression toolbox.

Created on Wed Jan 13 13:44:46 2016

@author: Maxime Sangnier
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from lightning.regression import QRegressor, toy_data


if __name__ == '__main__':
    probs = np.linspace(0.1, 0.9, 5)  # Joint quantile regression
    # probs = [0.5]  # Single quantile regression
    algorithms = ['qp', 'al', 'sdca']  # Algorithms to compare

    x_train, y_train, z_train = toy_data(50)
    x_test, y_test, z_test = toy_data(1000, t_min=-0.2, t_max=1.7, probs=probs)
    reg = QRegressor(C=1e2, probs=probs, gamma_out=1e-2, max_iter=1e4, verbose=False)

    res = []  # List for resulting coefficients
    plt.figure(figsize=(12, 7))
    for it, alg in enumerate(algorithms):
        reg.alg = alg

        # Fit on training data and predict on test data
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)

        # Plot the estimated conditional quantiles
        plt.subplot(1, len(algorithms), it+1)
        plt.plot(x_train, y_train, '.')
        for q in pred:
           plt.plot(x_test, q, '-')
        for q in z_test:
           plt.plot(x_test, q, '--')
        plt.title(alg.upper())

        # Print the optimal objective value
        print(alg.upper() + " objective value: %f" % reg.obj)
        print(alg.upper() + " training time: %0.2fs" % reg.time)

        # Save optimal objectives and coefficients
        res.append((reg.obj, reg.coefs))

    # Comparison SDCA / CVXOPT / Augmented Lagrangian
    plt.figure(figsize=(12, 7))
    plt.subplot2grid((1, len(algorithms)*2), (0, 0), colspan=len(algorithms))
    for alg, (obj, coefs) in zip(algorithms, res):
       # Plot the solutions of SDCA, CVXOPT and AL
       plt.plot(coefs.ravel())
    plt.legend([alg.upper() for alg in algorithms])
    plt.title('Dual coefs')
    plt.plot([0, coefs.size], [0, 0], ':')

    for it, (alg, (obj, coefs)) in enumerate(zip(algorithms, res)):
        # Plot the solutions of SDCA, CVXOPT and AL
        plt.subplot2grid((1, len(algorithms)*2), (0, len(algorithms)+it))
        plt.imshow(np.fabs(coefs.T))
        plt.title(alg.upper())

    plt.show()



