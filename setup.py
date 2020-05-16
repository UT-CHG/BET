#!/usr/bin/env python

# Copyright (C) 2014-2020 The BET Development Team

'''
The python script for building the BET package and subpackages.
'''
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='bet',
      version='3.0.0',
      description='A toolkit for data-consistent stochastic problems.',
      author='Steven Mattis',
      author_email='steve.a.mattis@gmail.com',
      license='GNU LGPL',
      url='https://github.com/UT-CHG/BET',
      packages=['bet',
                'bet.sampling',
                'bet.calculateP',
                'bet.postProcess',
                'bet.sensitivity'],
      install_requires=['matplotlib',
                        'pyDOE',
                        'numpy',
                        'scipy',
                        'pytest',
                        'mpi4py'])
