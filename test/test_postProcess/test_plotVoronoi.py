# Copyright (C) 2014-2015 The BET Development Team

"""
This module contains tests for :module:`bet.postProcess.plotVoronoi`.


Tests for Voronoi plotting.
"""

import unittest
import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as simpleFunP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotVoronoi as plotVoronoi
import numpy as np
import scipy.spatial as spatial
import numpy.testing as nptest
import bet.util as util
from bet.Comm import comm
import os
import bet.sample as sample

@unittest.skipIf(comm.size > 1, 'Only run in serial')
class Test_plot_1D_voronoi(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotVoronoi.plot_1D_voronoi`
    for a 1D parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        emulated_input_samples = sample.sample_set(1)
        emulated_input_samples.set_domain(np.array([[0.0, 1.0]]))

        num_samples=100

        emulated_input_samples.set_values_local(np.linspace(emulated_input_samples.get_domain()[0][0],
                                             emulated_input_samples.get_domain()[0][1],
                                             num_samples+1))

        emulated_input_samples.set_probabilities_local(1.0/float(comm.size)*(1.0/float(\
                emulated_input_samples.get_values_local().shape[0]))\
                *np.ones((emulated_input_samples.get_values_local().shape[0],)))
        emulated_input_samples.set_volumes_local(1.0/float(comm.size)*(1.0/float(\
                emulated_input_samples.get_values_local().shape[0]))\
                *np.ones((emulated_input_samples.get_values_local().shape[0],)))
        emulated_input_samples.check_num()

        self.samples = emulated_input_samples
        
    def test_plot(self):
        """
        Test :meth:`bet.postProcess.plotVoronoi.plot_1D_voronoi`
        for a 1D parameter space.
        """
        try:
            plotVoronoi.plot_1D_voronoi(self.samples,
                                  filename = "file", interactive=False)
            go = True
            if os.path.exists("file.png"):
                os.remove("file.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

@unittest.skipIf(comm.size > 1, 'Only run in serial')
class Test_plot_2D_voronoi(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.plot_2D_voronoi`
    for a 2D parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        emulated_input_samples = sample.sample_set(2)
        emulated_input_samples.set_domain(np.array([[0.0,1.0],[0.0,1.0]]))

        emulated_input_samples.set_values_local(util.meshgrid_ndim((np.linspace(emulated_input_samples.get_domain()[0][0],
            emulated_input_samples.get_domain()[0][1], 10),
            np.linspace(emulated_input_samples.get_domain()[1][0],
                emulated_input_samples.get_domain()[1][1], 10))))

        emulated_input_samples.set_probabilities_local(1.0/float(comm.size)*\
                (1.0/float(emulated_input_samples.get_values_local().shape[0]))*\
                np.ones((emulated_input_samples.get_values_local().shape[0],)))
        emulated_input_samples.set_volumes_local(1.0/float(comm.size)*\
                (1.0/float(emulated_input_samples.get_values_local().shape[0]))*\
                np.ones((emulated_input_samples.get_values_local().shape[0],)))
        emulated_input_samples.check_num()

        self.samples = emulated_input_samples
   
        
    def test_plot(self):
        """
        Test :meth:`bet.postProcess.plotP.plot_2D_voronoi`
        for a 2D parameter space.
        """
        try:
            plotVoronoi.plot_2D_voronoi(self.samples,
                                  filename = "file", interactive=False)
            go = True
            if os.path.exists("file.png"):
                os.remove("file.png")
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)
