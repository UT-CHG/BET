#!/usr/bin/env python
'''
The python script for building the BET package and subpackages.
'''
from distutils.core import setup

setup(name='bet',
      version='0.0',
      description='Butler, Estep, Tavener method',
      author = 'Steven Mattis',
      author_email ='steve.a.mattis@gmail.com',
      url= 'https://github.com/smattis/BET',
      packages =['bet', 'bet.sampling', 'bet.calculateP', 'bet.postProcess'])
