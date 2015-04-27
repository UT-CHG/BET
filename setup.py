#!/usr/bin/env python
'''
The python script for building the BET package and subpackages.
'''
from distutils.core import setup

setup(name='bet',
      version='0.0',
      description='Butler, Estep, Tavener method',
      author='Steven Mattis',
      author_email='steve.a.mattis@gmail.com',
      liscense='GNU LGPL',
      url='https://github.com/UT-CHG/BET',
      packages=['bet', 'bet.sampling', 'bet.calculateP', 'bet.postProcess'],
      install_requires=['matplotlib', 'mpl_toolkits', 'pyDOE', 'scipy',
          'numpy'])
