import os.path

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('qreg', parent_package, top_path)

    srcdir = os.path.join(top_path, "qreg/")
    print(srcdir)

    config.add_extension('dataset_fast',
                         sources=['dataset_fast.cpp'],
                         include_dirs=[numpy.get_include(), srcdir])

    config.add_extension('sdca_qr_fast',
                         sources=['sdca_qr_fast.cpp'],
                         include_dirs=[numpy.get_include(), srcdir])

    config.add_extension('sdca_qr_al_fast',
                         sources=['sdca_qr_al_fast.cpp'],
                         include_dirs=[numpy.get_include(), srcdir])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
