# Lindley Graham 04/09/2015

"""
This module contains tests for :module:`bet.calculateP.simpleFunP`

Some of these tests make sure certain values are within a tolerance rather than
exact due to the stochastic nature of the algorithms being tested
"""

import os, bet, unittest
import bet.calculateP.simpleFunP as simpleFunP
import numpy as np

local_path = os.path.join(os.path.dirname(bet.__file__),
'../test/test_calulateP')

class prob_exact:
    def test_prob_sum_to_1(self):
        """
        Test to see if the 
