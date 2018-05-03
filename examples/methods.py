# encoding: utf-8
# Author: Maxime Sangnier
# License: BSD

"""
Quantile regression with operator-valued kernels and multi-task learning.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from qreg import QRegressor, QRegMTL, toy_data


if __name__ == '__main__':
    probs = np.linspace(0.1, 0.9, 5)  # Joint quantile regression
    x_train, y_train, z_train = toy_data(50)
    x_test, y_test, z_test = toy_data(1000, probs=probs)

    # QR with operator-valued kernel
    ovk = QRegressor(C=1e2, probs=probs, gamma_out=1e-2, alg='qp')

    # Fit on training data and predict on test data
    print("Learn QRegressor")
    ovk.fit(x_train, y_train)
    pred = ovk.predict(x_test)

    # Plot the estimated conditional quantiles
    plt.close('all')
    plt.figure(figsize=(12, 7))
    plt.subplot(231)
    plt.plot(x_train, y_train, '.')
    for q in pred:
        plt.plot(x_test, q, '-')
    for q in z_test:
        plt.plot(x_test, q, '--')
    plt.title('Operator-valued kernel')

    # QR with multi-task learning
    mtl = QRegMTL(C=1e2, probs=probs, n_landmarks=0.2, Creg=1e-12)

    # Fit on training data and predict on test data
    print("Learn QRegMTL (with location)")
    mtl.fit(x_train, y_train)
    pred = mtl.predict(x_test)

    # Plot the estimated conditional quantiles
    plt.subplot(232)
    plt.plot(x_train, y_train, '.')
    for q in pred:
        plt.plot(x_test, q, '-')
    for q in z_test:
        plt.plot(x_test, q, '--')
    plt.title('Multi-task learning (with location)')

    plt.subplot(235)
    plt.imshow(mtl.reg.D)
    plt.colorbar()
    plt.title('Learned metric (with location)')

    # QR with multi-task learning (without location regression)
    mtl = QRegMTL(C=1e4, probs=probs, n_landmarks=0.2, location=False)

    # Fit on training data and predict on test data
    print("Learn QRegMTL (without location)")
    mtl.fit(x_train, y_train)
    pred = mtl.predict(x_test)

    # Plot the estimated conditional quantiles
    plt.subplot(233)
    plt.plot(x_train, y_train, '.')
    for q in pred:
        plt.plot(x_test, q, '-')
    for q in z_test:
        plt.plot(x_test, q, '--')
    plt.title('Multi-task learning (without location)')

    plt.subplot(236)
    plt.imshow(mtl.reg.D)
    plt.colorbar()
    plt.title('Learned metric (without location)')

    # QR with multi-task learning (several parameters)
    Cs = np.logspace(-8, 8, num=8)
    plt.figure()
    for i, C in enumerate(Cs):
        print('Learn QRegMTL with C={}'.format(C))
        mtl = QRegMTL(C=C, probs=probs, n_landmarks=0.2, location=False)
        mtl.fit(x_train, y_train)
        pred = mtl.predict(x_test)

        # Plot the estimated conditional quantiles
        plt.subplot(4, 4, 4*(i//4)+i+1)
        plt.plot(x_train, y_train, '.')
        for q in pred:
            plt.plot(x_test, q, '-')
        for q in z_test:
            plt.plot(x_test, q, '--')
        plt.title('C={}'.format(C))

        plt.subplot(4, 4, 4*(i//4+1)+i+1)
        plt.imshow(mtl.reg.D)
        plt.colorbar()

    plt.show()

