#! /usr/bin/env python
#
# Copyright (C) 2012 Maxime Sangnier, Olivier Fercoq

import sys
import os
import setuptools
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('qreg')

    return config

if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name='qreg',
          maintainer='Maxime Sangnier',
          maintainer_email='maxime.sangnier@upmc.fr',
          description='Data sparse quantile regression in Python',
          license='New BSD',
          url='https://github.com/msangnier/qreg',
          version='0.1',
          download_url='https://github.com/msangnier/qreg',
          long_description=open('README.rst').read(),
          zip_safe=False,
          install_requires=['numpy', 'cvxopt'])
