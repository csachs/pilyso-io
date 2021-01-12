# -*- coding: utf-8 -*-
"""
documentation
"""

from setuptools import setup, find_packages

setup(
    name='pilyso_io',
    version='0.0.1.dev4',
    description='pilyso - image reading library',
    long_description='Python Image anaLYsis SOftware library',
    author='Christian Sachs',
    author_email='c.sachs@fz-juelich.de',
    url='https://github.com/modsim/pilyso-io',
    packages=find_packages(),
    install_requires=['numpy', 'nd2file', 'tifffile', 'czifile'],
    extras_require={
        'ndip': ['opencv']
    },
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ]
)
