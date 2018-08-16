.. -*- mode: rst -*-

qreg
====

qreg is a Python library for data sparse and non-parametric quantile regression. It implements quantile regression with matrix-valued kernels and makes it possible to learn several quantile curves simultaneously with a sparsity requirement on supporting data.

Highlights:

- based on the library `lightning <https://github.com/mblondel/lightning>`_;
- follows the `scikit-learn <http://scikit-learn.org>`_ style of programming;
- computationally demanding parts implemented in `Cython <http://cython.org>`_.

Example
-------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from qreg import QRegressor, toy_data
    
    # Quantile levels to prediect
    probs = np.linspace(0.1, 0.9, 5)
    
    # Train and test dataset
    x_train, y_train, z_train = toy_data(50)
    x_test, y_test, z_test = toy_data(1000, t_min=-0.2, t_max=1.7, probs=probs)
    
    # Define the quantile regressor
    reg = QRegressor(C=1e2,  # Trade-off parameter
                     probs=probs,  # Quantile levels
                     gamma_out=1e-2,  # Inner kernel parameter
                     eps=2,  # Epsilon-loss level
                     alg='sdca',  # Algorithm (can change to 'qp')
                     max_iter=1e4,  # Maximal number of iteration
                     active_set=True,  # Active set strategy
                     verbose=True)
    
    # Fit on training data and predict on test data
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    
    # Plot the estimated conditional quantiles
    plt.plot(x_train, y_train, '.')
    for q in pred:
        plt.plot(x_test, q, '-')
    for q in z_test:
        plt.plot(x_test, q, '--')
    
    # Print some information
    print("Objective value: %f" % reg.obj)
    print("Training time: %0.2fs" % reg.time)
    print("#SV: %d" % reg.num_sv())
    print("Score: %f" % reg.score(x_test, y_test))
    
    plt.show()

Dependencies
------------

qreg needs Python >= 2.7, setuptools, Numpy, SciPy, scikit-learn, cvxopt and a working C/C++ compiler.

Installation
------------

To install qreg from pip, type::

    pip install https://github.com/msangnier/qreg/archive/master.zip

To install qreg from source, type::

    git clone https://github.com/msangnier/qreg.git
    cd qreg
    python setup.py build
    sudo python setup.py install

Authors
-------

Olivier Fercoq and Maxime Sangnier

References
----------

- Data sparse nonparametric regression with epsilon-insensitive losses (2017), M. Sangnier, O. Fercoq, F. d'Alché-Buc. Asian Conference on Machine Learning (ACML).
- Joint quantile regression in vector-valued RKHSs (2016), M. Sangnier, O. Fercoq, F. d'Alché-Buc. Neural Information Processing Systems (NIPS).

