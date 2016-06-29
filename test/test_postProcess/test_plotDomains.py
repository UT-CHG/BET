# Copyright (C) 2014-2016 The BET Development Team

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
from bet.Comm import comm
import bet.sample as sample

local_path = os.path.join(os.path.dirname(bet.__file__),
        "../test/test_sampling")

@unittest.skipIf(comm.size > 1, 'Only run in serial')
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
        # Create sample_set object for input_samples
        input_samples = sample.sample_set(4)

        input_samples.set_domain(np.array([[0.0, 1.0], [0.0, 1.0],
                                           [0.0, 1.0], [0.0, 1.0]]))
        input_samples.set_values(util.meshgrid_ndim(
            (np.linspace(input_samples.get_domain()[0,0],
            input_samples.get_domain()[0,1], 3),
             np.linspace(input_samples.get_domain()[1,0],
            input_samples.get_domain()[1,1], 3),
             np.linspace(input_samples.get_domain()[2,0],
            input_samples.get_domain()[2,1], 3),
             np.linspace(input_samples.get_domain()[3,0],
            input_samples.get_domain()[3,1], 3))))
        input_samples.set_probabilities(
            (1.0/float(input_samples.get_values().shape[0]))
            *np.ones((input_samples.get_values().shape[0],)))

        input_samples.check_num() # Check that probabilities and values arrays have same number of entries

        # Create sample_set object for output_samples
        output_samples = sample.sample_set(4)
        output_samples.set_values(input_samples.get_values()*3.0)
        output_samples.set_domain(3.0*input_samples.get_domain())

        self.disc = sample.discretization(input_samples, output_samples)

        self.filename = "testfigure"

        output_ref_datum = np.mean(output_samples.get_domain(), axis=1)

        bin_size = 0.15*(np.max(output_samples.get_domain(), axis=1) -
                         np.min(output_samples.get_domain(), axis=1))
        maximum = 1/np.product(bin_size)

        def ifun(outputs):
            """
            Indicator function.
            :param outputs: outputs
            :type outputs: :class:`numpy.ndarray` of shape (N, ndim)
            :rtype: :class:`numpy.ndarray` of shape (N,)
            :returns: 0 if outside of set or positive number if inside set
            """
            left = np.repeat([output_ref_datum-.5*bin_size], outputs.shape[0], 0)
            right = np.repeat([output_ref_datum+.5*bin_size], outputs.shape[0], 0)
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
        filenames.extend(glob.glob('param_samples_*cs.*'))
        filenames.extend(glob.glob('data_samples_*cs.*'))

        filenames.extend(glob.glob(self.filename+".*"))
        filenames.extend(glob.glob( 'param_samples_*cs.*'))
        filenames.extend(glob.glob(os.path.join(local_path,
            'data_samples_*cs.*')))


        filenames.append('domain_q1_q2_cs.*')
        filenames.append('domain_q1_q1_cs.*')
        filenames.append('q1_q2_domain_Q_cs.*')
        filenames.append('q1_q1_domain_Q_cs.*')
        figfiles = glob.glob('figs/*')
        figfiles.extend(glob.glob(os.path.join(local_path, 'figs/*')))
        figfiles.extend(glob.glob(os.path.join(local_path, '*.png')))
        figfiles.extend(glob.glob('*.png'))
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
        p_ref = [None, self.disc._input_sample_set.get_values()[4, [0, 1]]]
        #p_ref = [None, self.samples[4, [0, 1]]]
        for sn, pr in zip(sample_nos, p_ref):
            self.check_scatter_2D(sn, pr, True)

    def check_scatter_2D(self, sample_nos, p_ref, save):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.scatter_2D` ran
        without generating an error.
        """
        try:
            input_sample_set_temp = sample.sample_set(2)
            input_sample_set_temp.set_values(self.disc._input_sample_set.get_values()[:, [0, 1]])
            plotDomains.scatter_2D(
                input_sample_set_temp,
                sample_nos,
                self.disc._input_sample_set.get_probabilities(),
                p_ref, save, False, 'XLABEL', 'YLABEL', self.filename)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False

        nptest.assert_equal(go, True)

    def test_scatter_3D(self):
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_3D`
        """
        sample_nos = [None, 25]
        p_ref = [None, self.disc._input_sample_set.get_values()[4, :]]
        for sn, pr in zip(sample_nos, p_ref):
                self.check_scatter_3D(sn, pr, True)

    def check_scatter_3D(self, sample_nos, p_ref, save):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.scatter_3D` ran
        without generating an error.
        """
        try:
            input_sample_set_temp = sample.sample_set(3)
            input_sample_set_temp.set_values(self.disc._input_sample_set.get_values()[:, [0, 1, 2]])
            plotDomains.scatter_3D(
                input_sample_set_temp,
                sample_nos,
                self.disc._input_sample_set.get_probabilities(),
                p_ref, save, False, 'XLABEL', 'YLABEL', 'ZLABEL', self.filename)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False

        nptest.assert_equal(go, True)

    def test_show_param(self):
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_rhoD`
        """
        sample_nos = [None, 25]
        samples = [self.disc._input_sample_set.get_values(),
                   self.disc._input_sample_set.get_values()[:, [0, 1]],
                   self.disc._input_sample_set.get_values()[:, [0, 1, 2]]]
        lnums = [None, self.lnums]

        for sample in samples:
            showdim = [None]
            if sample.shape[0] > 2:
                showdim.append(2)
            if sample.shape[0] > 3:
                showdim.append(3)
            for sd in showdim:
                p_ref = [None, sample[4, :]]
                for ln, sn, pr in zip(lnums, sample_nos, p_ref):
                    self.check_show_param(sample, sn, pr, True, ln, sd)

    def check_show_param(self, samples, sample_nos, p_ref, save, lnums,
            showdim):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.scatter_rhoD` ran
        without generating an error.
        """
        try:
            input_sample_set_temp = sample.sample_set(samples.shape[1])
            input_sample_set_temp.set_values(samples)
            disc_obj_temp = sample.discretization(input_sample_set_temp,
                                                  self.disc._output_sample_set)
            plotDomains.scatter_rhoD(disc_obj_temp, p_ref, sample_nos, 'input',
                    self.rho_D, lnums, None, showdim, save, False)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False

        nptest.assert_equal(go, True)

    def test_show_data(self):
        """
        Test :meth:`bet.postProcess.plotDomains.scatter_rhoD`
        """
        sample_nos = [None, 25]
        data_sets = [self.disc._output_sample_set.get_values(),
                     self.disc._output_sample_set.get_values()[:, [0, 1]]]
        qnums = [None, [0, 1, 2]]#self.lnums]

        for data, qn, sn in zip(data_sets, qnums, sample_nos):
            showdim = [None]
            if data.shape[0] > 2:
                showdim.append(2)
            if data.shape[0] > 3:
                showdim.append(3)
            Q_ref = [None, data[4, :]]
            for sd, qr in zip(showdim, Q_ref):
                self.check_show_data(data, sn, qr, True, qn, sd)

    def check_show_data(self, data, sample_nos, q_ref, save, qnums, showdim):
        """
        Check to see that the :meth:`bet.postTools.plotDomains.scatter_rhoD` ran
        without generating an error.
        """
        try:
            if data.shape[1] == 4:
                data_obj_temp = sample.sample_set(4)
                data_obj_temp.set_values(data)
                plotDomains.scatter_rhoD(data_obj_temp, q_ref, sample_nos,
                        'output', self.rho_D, qnums, None, showdim, save,
                        False) 
            else:
                data_obj_temp = sample.sample_set(data.shape[1])
                data_obj_temp.set_values(data)
                plotDomains.scatter_rhoD(data_obj_temp, q_ref, sample_nos,
                        None, None, qnums, None, showdim, save, False) 
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
        filenames = [None, ['domain_q1_q1_cs', 'q1_q2_domain_Q_cs']]

        for rm, rc, fn in zip(ref_markers, ref_colors, filenames):
            self.check_show_data_domain_2D(rm, rc, None, True, fn)

    def check_show_data_domain_2D(self, ref_markers, ref_colors, triangles,
            save, filenames):
        """
        Check to see that the
        :meth:`bet.postTools.plotDomains.show_data_domain_2D` ran
        without generating an error.
        """
        Q_ref = self.disc._output_sample_set.get_values()[:, [0, 1]]
        Q_ref = Q_ref[[1,4],:]

        data_obj_temp = sample.sample_set(2)
        data_obj_temp.set_values(self.disc._output_sample_set.get_values()[:, [0, 1]])
        disc_obj_temp = sample.discretization(self.disc._input_sample_set,data_obj_temp)

        try:
            plotDomains.show_data_domain_2D(
                disc_obj_temp, Q_ref,
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

        for rm, rc in zip(ref_markers, ref_colors):
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
        Q_ref = self.disc._output_sample_set.get_values()[[4, 2], :]
        try:
            plotDomains.show_data_domain_multi(
                self.disc,
                Q_ref, Q_nums, ref_markers=ref_markers,
                ref_colors=ref_colors, showdim=showdim)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False
        nptest.assert_equal(go, True)      

    def test_scatter_2D_multi(self):
        """
        Test :meth:`bet.postTools.plotDomins.scatter_2D_multi`
        """
        if not os.path.exists('figs/'):
            os.mkdir('figs/')
        try:
            input_sample_set_temp = sample.sample_set(3)
            input_sample_set_temp.set_values(self.disc._input_sample_set.get_values()[:, [0,1,2]])

            plotDomains.scatter_2D_multi(input_sample_set_temp)
            go = True
        except (RuntimeError, TypeError, NameError):
            go = False

        nptest.assert_equal(go, True)

