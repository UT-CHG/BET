# Lindley Graham 04/07/2015
"""
This module contains tests for :module:`bet.postProcess.plotDomains`.


Tests for the execution of plotting parameter and data domains.
"""

import unittest, os, glob, bet
import bet.postProcess.plotDomains as plotDomains
import bet.util as util
import matplotlib.tri as tri
from matplotlib.lines import Line2D
import numpy as np
import numpy.testing as nptest
from bet.Comm import size

local_path = os.path.join(os.path.dirname(bet.__file__),
        "../test/test_sampling")

@unittest.skipIf(size > 1, 'Only run in serial')
class test_plotDomains(unittest.TestCase):
    """
    Test :meth:`bet.postProcess.plotP.calculate_1D_marginal_probs` and  
    :meth:`bet.postProcess.plotP.calculate_2D_marginal_probs` for a 2D
    parameter space.
    """
    def setUp(self):
        """
        Set up problem.
        """
        self.lam_domain = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        self.samples = util.meshgrid_ndim((np.linspace(self.lam_domain[0][0],
            self.lam_domain[0][1], 10), np.linspace(self.lam_domain[1][0],
            self.lam_domain[1][1], 10), np.linspace(self.lam_domain[1][0],
            self.lam_domain[1][1], 10), np.linspace(self.lam_domain[1][0],
            self.lam_domain[1][1], 10)))
        self.data = self.samples*3.0
        self.P_samples = (1.0/float(self.samples.shape[0]))*np.ones((self.samples.shape[0],))
        self.filename = "testfigure"

        QoI_range = np.array([3.0, 3.0, 3.0, 3.0])
        Q_ref = QoI_range*0.5
        bin_size = 0.15*QoI_range
        maximum = 1/np.product(bin_size)
        def ifun(outputs):
            """
            Indicator function.
            :param outputs: outputs
            :type outputs: :class:`numpy.ndarray` of shape (N, ndim)
            :rtype: :class:`numpy.ndarray` of shape (N,)
            :returns: 0 if outside of set or positive number if inside set
            """
            left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
            right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
            left = np.all(np.greater_equal(outputs, left), axis=1)
            right = np.all(np.less_equal(outputs, right), axis=1)
            inside = np.logical_and(left, right)
            max_values = np.repeat(maximum, outputs.shape[0], 0)
            return inside.astype('float64')*max_values
        self.rho_D = ifun
        self.lnums = [1, 2, 3]
        self.markers = []
        for m in Line2D.markers:
            try:
                if len(m) == 1 and m != ' ':
                    self.markers.append(m)
            except TypeError:
                pass
        self.colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    def tearDown(self):
        """
        Tear Down problem
        """
        # remove any files the we create
        filenames = glob.glob(self.filename+".*")
        filenames.extend(glob.glob('param_samples_*cs.eps'))
        filenames.extend(glob.glob('data_samples_*cs.eps'))

        filenames.extend(glob.glob(self.filename+".*"))
        filenames.extend(glob.glob( 'param_samples_*cs.eps'))
        filenames.extend(glob.glob(os.path.join(local_path,
            'data_samples_*cs.eps')))


        filenames.append('domain_q1_q2_cs.eps')
        filenames.append('domain_q1_q1_cs.eps')
        filenames.append('q1_q2_domain_Q_cs.eps')
        filenames.append('q1_q1_domain_Q_cs.eps')
        figfiles = glob.glob('figs/*')
        figfiles.extend(glob.glob(os.path.join(local_path, 'figs/*')))
        filenames.extend(figfiles)

        for f in filenames:
            if os.path.exists(os.path.join(local_path, f)):
                os.remove(os.path.join(local_path, f))
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("figs"):
            os.rmdir("figs")

    def test_scatter_2D(self):
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_2D`
        """
        sample_nos = [None, 25]
        p_ref = [None, self.samples[4, [0, 1]]]
        save = [True, False]
        for sn in sample_nos:
            for pr in p_ref:
                for s in save:
                    self.check_scatter_2D(sn, pr, s)

    def check_scatter_2D(self, sample_nos, p_ref, save):
        """

        Check to see that the :meth:`bet.postTools.plotDomains.scatter_2D` ran
        without generating an error.
        """
        try:
            plotDomains.scatter_2D(self.samples[:, [0, 1]], sample_nos,
                    self.P_samples, p_ref, save, False, 'XLABEL', 'YLABEL',
                    self.filename)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)

    def test_scatter_3D(self):
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_3D`
        """
        sample_nos = [None, 25]
        p_ref = [None, self.samples[4, :]]
        save = [True, False]
        for sn in sample_nos:
            for pr in p_ref:
                for s in save:
                    self.check_scatter_3D(sn, pr, s)

    def check_scatter_3D(self, sample_nos, p_ref, save):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.scatter_3D` ran
        without generating an error.
        """
        try:
            plotDomains.scatter_3D(self.samples[:, [0, 1, 2]], sample_nos,
                    self.P_samples, p_ref, save, False, 'XLABEL', 'YLABEL',
                    'ZLABEL', self.filename) 
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)      

    def test_show_param(self):
        """
        Test :meth:`bet.postProcess.plotDomains.show_param`
        """
        sample_nos = [None, 25]
        save = [True, False]
        samples = [self.samples, self.samples[:, [0, 1]],
                self.samples[:, [0, 1, 2]]]
        lnums = [None, self.lnums]

        for sample in samples:
            showdim = [None]
            if sample.shape[0] > 2:
                showdim.append(2)
            if sample.shape[0] > 3:
                showdim.append(3)
            for sd in showdim:
                for ln in lnums:
                    for sn in sample_nos:
                        p_ref = [None, sample[4, :]]
                        for pr in p_ref:
                            for s in save:
                                self.check_show_param(sample, sn, pr, s, ln,
                                        sd)

    def check_show_param(self, samples, sample_nos, p_ref, save, lnums,
            showdim):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.show_param` ran
        without generating an error.
        """
        try:
            plotDomains.show_param(samples, self.data, self.rho_D, p_ref,
                    sample_nos, save, False, lnums, showdim) 
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True) 

    def test_show_data(self):
        """
        Test :meth:`bet.postProcess.plotDomains.show_data`
        """
        sample_nos = [None, 25]
        save = [True, False]
        data_sets = [self.data, self.data[:, [0, 1]]]
        qnums = [None, [0, 1, 2]]#self.lnums]

        for data in data_sets:
            showdim = [None]
            if data.shape[0] > 2:
                showdim.append(2)
            if data.shape[0] > 3:
                showdim.append(3)
            for sd in showdim:
                for qn in qnums:
                    for sn in sample_nos:
                        Q_ref = [None, data[4, :]]
                        for qr in Q_ref:
                            for s in save:
                                self.check_show_data(data, sn, qr, s, qn, sd)

    def check_show_data(self, data, sample_nos, q_ref, save, qnums, showdim):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.show_data` ran
        without generating an error.
        """
        try:
            if data.shape[1] == 4:
                plotDomains.show_data(data, self.rho_D, q_ref,
                    sample_nos, save, False, qnums, showdim) 
            else:
                plotDomains.show_data(data, None, q_ref,
                    sample_nos, save, False, qnums, showdim) 
            go = True
        except (RuntimeError, TypeError, NameError):
            print "ERROR"
            print data.shape
            print q_ref
            print sample_nos
            print save
            print qnums
            print showdim
            go = False
        nptest.assert_equal(go, True)

    def test_show_data_domain_2D(self):
        """
        Test :meth:`bet.postProces.plotDomains.show_data_domain_2D`
        """
        ref_markers = [None, self.markers]
        ref_colors = [None, self.colors]
        triangulation = tri.Triangulation(self.samples[:, 0], self.samples[:, 1])
        triangles = [None, triangulation.triangles]
        filenames = [None, ['domain_q1_q1_cs.eps', 'q1_q2_domain_Q_cs.eps']]
        save = [None, False]

        for rm in ref_markers:
            for rc in ref_colors:
                for t in triangles:
                    for s in save:
                        for fn in filenames:
                            self.check_show_data_domain_2D(rm, rc, t, s, fn)

    def check_show_data_domain_2D(self, ref_markers, ref_colors, triangles,
            save, filenames):
        """
        Check to see that the
        :meth:`bet.postTools.plotDomains.show_data_domain_2D` ran
        without generating an error.
        """
        Q_ref = self.data[:, [0, 1]]
        Q_ref = Q_ref[[1,4],:]
        print Q_ref.shape
        data = self.data[:, [0, 1]]
        try:
            plotDomains.show_data_domain_2D(self.samples, data, Q_ref,
                    ref_markers, ref_colors, triangles=triangles, save=save,
                    filenames=filenames)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True) 

    def test_show_data_domain_multi(self):
        """
        Test :meth:`bet.postProcess.plotDomains.show_data_domain_multi`
        """
        if not os.path.exists('figs/'):
            os.mkdir('figs/')
        Q_nums = [None, [1, 2], [1, 2, 3]]
        ref_markers = [None, self.markers]
        ref_colors = [None, self.colors]
        for rm in ref_markers:
            for rc in ref_colors:
                for qn in Q_nums:
                    showdim = [None, 1]
                    if qn and len(qn) > 2:
                        showdim.extend(['all', 'ALL'])
                    for sd in showdim:
                        self.check_show_data_domain_multi(rm, rc, qn, sd)

    def check_show_data_domain_multi(self, ref_markers, ref_colors, Q_nums,
            showdim):
        """
        Check to see that the
        :meth:`bet.postTools.plotDomains.show_data_domain_multi` ran
        without generating an error.
        """
        Q_ref = self.data[[4, 2], :]
        try:
            plotDomains.show_data_domain_multi(self.samples, self.data,
                    Q_ref, Q_nums, ref_markers=ref_markers,
                    ref_colors=ref_colors, showdim=showdim)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)      

