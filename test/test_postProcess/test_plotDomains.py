# Lindley Graham 04/07/2015
"""
This module contains tests for :module:`bet.postProcess.plotDomains`.


Tests for the execution of plotting parameter and data domains.
"""

import unittest, os, glob
import bet.postProcess.plotD as plotDomains
import numpy as np
import numpy.testing as nptest
from bet.Comm import *

class Test_calc_marg_2D(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs` and  :meth:`bet.postProcess.plotP.calculate_2D_marginal_probs` for a 2D
    parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.lam_domain=np.array([[0.0,1.0],[0.0,1.0], [0.0, 1.0]])
        self.samples=util.meshgrid_ndim((np.linspace(self.lam_domain[0][0],
            self.lam_domain[0][1], 10),np.linspace(self.lam_domain[1][0],
                self.lam_domain[1][1], 10), np.linspace(self.lam_domain[1][0],
                    self.lam_domain[1][1], 10)))
        self.data = self.samples*3.0
        self.P_samples = (1.0/float(self.samples.shape[0]))*np.ones((self.samples.shape[0],))
        self.filename("testfigure")
        
        QoI_range = np.array([3.0, 3.0, 3.0])
        Q_ref = QoI_range*0.5
	bin_size = 0.15*QoI_range
	maximum = 1/np.product(bin_size)
	def ifun(outputs):
	    left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
	    right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
	    left = np.all(np.greater_equal(outputs, left), axis=1)
	    right = np.all(np.less_equal(outputs, right), axis=1)
	    inside = np.logial_and(left, right)
	    max_values = np.repeate(maximum, outputs.shape[0], 0)
	    return inside.astype('float64')*max_values
	self.rho_D = ifun
        self.lnums = [1,2,3]

    def tearDown(self):
        """
        Tear Down problem
        """
        # remove any files the we create
        filenames = glob.glob(self.filename+".*")
        filenames.append('param_samples_cs.eps')
        for f in filenames:
            os.remove(f)

    def test_scatter_2D():
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_2D`
        """
        sample_nos = [None, 25]
        p_ref = [None, self.samples[4,[0, 1]]]
        save = [True, False]
        for sn in sample_nos:
            for pr in p_ref:
                for s in save:
                    yield self.check_scatter_2D, sn, pr, s

    def check_scatter_2D(self, sample_nos, p_ref, save):
        try:
            plotDomains.scatter_2D(self.samples[:,[0, 1]], sample_nos,
                self.P_samples, p_ref, save, False, 'XLABEL', 'YLABEL',
                self.filename)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)
 
    def test_scatter_3D():
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_3D`
        """
        sample_nos = [None, 25]
        p_ref = [None, self.samples[4,:]]
        save = [True, False]
        for sn in sample_nos:
            for pr in p_ref:
                for s in save:
                    yield self.check_scatter_3D, sn, pr, s

    def check_scatter_3D(self, sample_nos, p_ref, save):
        try:
            plotDomains.scatter_3D(self.samples, sample_nos, self.P_samples,
                    p_ref, save, False, 'XLABEL', 'YLABEL', 'ZLABEL',
                    self.filename) 
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)      

    def test_show_param():
        """
        Test :meth:`bet.postProcess.plotDomains.show_param`
        """
        sample_nos = [None, 25]
        save = [True, False]
        samples = [self.samples, self.samples[:,[0, 1]]]
        lnums = [None, self.lnums]

        for sample in samples:
            for ln in lnums:
                for sn in sample_nos:
                    p_ref = [None, samples[4,:]]
                    for pr in p_ref:
                        for s in save:
                            yield self.check_show_param, samples, pr, sn, s, ln

    def check_show_param(self, samples, sample_nos, p_ref, save, lnums):
        try:
            plotDomains.show_param(samples, self.data, self.rho_D, p_ref,
                    sample_nos, save, False, lnums) 
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True) 

    def test_show_data():
        """
        Test :meth:`bet.postProcess.plotDomains.show_data`
        """
        sample_nos = [None, 25]
        save = [True, False]
        data_sets = [self.data, self.data[:,[0, 1]]]
        qnums = [None, self.lnums]

        for data in data_sets:
            for qn in qnums:
                for sn in sample_nos:
                    Q_ref = [None, data[4,:]]
                    for qr in Q_ref:
                        for s in save:
                            yield self.check_show_data, data, sn, qr, s, qn

    def check_show_data(self, data, sample_nos, q_ref, save, qnums):
        try:
            plotDomains.show_data(data, self.rho_D, q_ref,
                    sample_nos, save, False, qnums) 
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True) 

