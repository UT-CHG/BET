# Lindley Graham 05/22/2014

"""
This module contains tests for :module:`bet.caculcateP.calculateP`.

    * compare using same lambda_emulated (iid and regular grid)
    * compare using different lambda_emulated

Most of these tests should make sure certain values are within a tolerance
rather than exact due to the stocastic nature of the algorithms being tested.
"""

import unittest
import bet.calculateP.calculateP as calcP
import numpy sa np
import scipy.spatial as spatial
import numpy.testing as nptest

class TestEmulateIIDLebesgue(unittest.TestCase):
    """
    Test :meth:`bet.calculateP.calculateP.emulate_iid_lebesgue`.
    """
    
    def runTest(self):
        """
        Test dimension, number of samples, and that all the samples are within
        lambda_domain.

        """
        lam_left = np.array([0.0, .25, .4])
        lam_right = np.array([1.0, 4.0, .5])

        lam_domain = np.zeros((3,3))
        lam_domain[:,0] = lam_left
        lam_domain[:,1] = lam_right

        num_l_emulate = 1e6

        lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)

        # check the dimension
        np.assert_array_equal(lambda_emulate.shape, (3, num_l_emulate))

        # check that the samples are all within the correct bounds
        np.assertGreaterEqual(0.0, np.min(lambda_emulate[0, :]))
        np.assertGreaterEqual(.25, np.min(lambda_emulate[1, :]))
        np.assertGreaterEqual(.4, np.min(lambda_emulate[2, :]))
        np.assertLessEqual(1.0, np.max(lambda_emulate[0, :]))
        np.assertLessEqual(4.0, np.max(lambda_emulate[1, :]))
        np.assertLessEqual(.5, np.max(lambda_emulate[2, :]))

# make sure probabilties and lam_vol follow the MC assumption (i.e. for uniform
# over the entire space they should all be the same, the hyperrectangle case
# will be different)

# compare the P calculated by the different methods on the same samples 

# add a skip thing to the tests involving qhull so that it only runs if that
# package is installed

# test where P is uniform over the entire domain
# test where P is uniform over a hyperrectangle subdomain
# test with and without optional arguments, together and separatly
# compare using same lambda_emulated (iid and regular grid)
# compare using different lambda_emulated
# some will need to be within a tolerance and some will need to be exact
# also tolerances will need to be depend on if the lambda_emulated are the same
# and if there voronoi cells are the same, etc.
# test on a linear model and maybe a non-linear, read from file model

class TestProb(object):
    """
    Tests ``prob*`` methods in :mod:`bet.calculateP.calculateP` with a linear
    model.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create inputs for ``prob*`` methods. This should run only once per
        grouping of tests from this class. But, for isolated tests it doesn't
        make any sense to do it this way. Since many of these tests are
        comparision tests it should be fine to structure things in this manner
        rather using inheritance everywhere.
        """
        # Create model associated inputs (samples, data, lam_domain)
        data_domain = np.array([[0.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]])
        self.lam_domain = np.array([[.1, .2], [3, 4], [50, 60]])
        # iid samples
        self.u_samples = None #calcP.emulate_iid_lebesgue(lam_domain,
                #20**lam_domain.shape[0])
        # regular grid samples
        lam1 = np.linspace(lam_domain[0, 0], lam_domain[0, 1], 20)
        lam2 = np.linspace(lam_domain[1, 0], lam_domain[1, 1], 20)
        lam3 = np.linspace(lam_domain[2, 0], lam_domain[2, 1], 20)
        lam1, lam2, lam3 = np.meshgrid(lam1, lam2, lam3)
        self.r_samples = np.column_stack((lam1.ravel(), lam2.ravel(), lam3.ravel()))
    
        # Create lambda_emulate
        # iid
        self.u_lambda_emulate = None #calcP.emulate_iid_lebesgue(lam_domain,
            1e6)
        # regular grid
        lam1 = np.linspace(lam_domain[0, 0], lam_domain[0, 1], 100)
        lam2 = np.linspace(lam_domain[1, 0], lam_domain[1, 1], 100)
        lam3 = np.linspace(lam_domain[2, 0], lam_domain[2, 1], 100)
        lam1, lam2, lam3 = np.meshgrid(lam1, lam2, lam3)
        self.r_lambda_emulate = np.column_stack((lam1.ravel(), lam2.ravel(),
            lam3.ravel()))

    def compare_to_vol(result_vol, lambda_domain):
        """
        Compare lambda_vol from the algorithm to an analytic solution.

        :param result_vol: lambda_vol from any of the methods in
            :mod:`~bet.calculatevol.calculatevol`
        :type result_vol: :class:`numpy.ndarray`

        """
        lambda_vol = np.product(lambda_domain[:,1]-lambda_domain[:,0])
        lambda_vol = lambda_vol / float(len(result_vol))
        nptest.assert_array_equal(result_vol,
                np.ones(result_vol.shape)*lambda_vol)
        
    def compare_to_vol_ae(result_vol, lambda_domain):
        """
        Compare ``lambda_vol`` from the algorithm to an analytic solution.

        :param result_vol: lambda_vol from any of the methods in
            :mod:`~bet.calculatevol.calculatevol`
        :type result_vol: :class:`numpy.ndarray`

        """
        lambda_vol = np.product(lambda_domain[:,1]-lambda_domain[:,0])
        lambda_vol = lambda_vol / float(len(result_vol))
        nptest.assert_array_almost_equal_nulp(result_vol,
                np.ones(result_vol.shape)*lambda_vol)

    def compare_prob_dtree(result_wtree, result_wotree):
        """

        Make sure the output from
        :meth:`bet.calcuateP.calculateP.prob_emulated` matches with and without
        option arguments.

        """
        # calculate with d_tree
        (P, lem, io_ptr, emulate_ptr) = result_wtree
        # calculate without d_tree
        (Pt, lemt, io_ptrt, emulate_ptrt) = result_wotree
        # Compare results
        nptest.assert_array_equal(P,Pt)
        nptest.assert_array_equal(lem, lemt)
        nptest.assert_array_equal(lem, self.r_lambda_emulate)
        nptest.assert_array_equal(io_ptrt, ioptr)
        nptest.assert_array_equal(emulate_ptr, emulate_ptrt)

    def compare_prob_emulate(result_wsamples, result_wosamples):
        """

        Make sure the output from
        :meth:`bet.calcuateP.calculateP.prob_emulated` matches with and without
        ``lambda_emulate`` when ``lambda_emulate == samples``.

        """
        # calculate with samples
        (P, lem, io_ptr, emulate_ptr) = result_wsamples
        # calculate without samples
        (Pt, lemt, io_ptrt, emulate_ptrt) = result_wosamples
        # Compare results
        nptest.assert_array_equal(P,Pt)
        nptest.assert_array_equal(lem, lemt)
        nptest.assert_array_equal(lem, self.r_lambda_emulate)
        nptest.assert_array_equal(io_ptrt,ioptr)
        nptest.assert_array_equal(emulate_ptr, emulate_ptrt)

    def compare_prob(result_emulated, result_prob, result_mc):
        """

        Make sure that the output from 
        :meth:`~bet.calculateP.calculateP.prob_emulated`,
        :meth:`~bet.calculateP.calculateP.prob`,
        :meth:`~bet.calculateP.calculateP.prob_mc` matches when ``lambda_emulate == samples``.

        .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.

        """
        # Calculate prob 
        (P, lem, io_ptr, emulate_ptr) = result_emulated
        (P1, lam_vol1, lem1, io_ptr1, emulate_ptr1) = result_prob
        (P3, lam_vol3, lem3, io_ptr3, emulate_ptr3) = result_mc

        # Compare results
        nptest.assert_array_equal(P,P1)
        nptest.assert_array_equal(P,P3)
        nptest.assert_array_equal(P1,P3)

        nptest.assert_array_equal(lam_vol1,lam_vol3)

        nptest.assert_array_equal(lem,lem1)
        nptest.assert_array_equal(lem,lem3)
        nptest.assert_array_equal(lem1,lem3)

        nptest.assert_array_equal(io_ptr,io_ptr1)
        nptest.assert_array_equal(io_ptr,io_ptr3)
        nptest.assert_array_equal(io_ptr1,io_ptr3)
       
        nptest.assert_array_equal(emulate_ptr,emulate_ptr1)
        nptest.assert_array_equal(emulate_ptr,emulate_ptr3)
        nptest.assert_array_equal(emulate_ptr1,emulate_ptr3)

    def compare_volume_rg(result_prob, result_mc):
        """

        Make sure that the voronoi cell volumes from 
        :meth:`~bet.calculateP.calculateP.prob`,
        :meth:`~bet.calculateP.calculateP.prob_mc` matches when the samples are
        all on a regular grid and ``lambda_emulate == samples``.

        .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.

        """
        # Calculate prob 
        (P1, lam_vol1, lem1, io_ptr1, emulate_ptr1) = result_prob
        (P3, lam_vol3, lem3, io_ptr3, emulate_ptr3) = result_mc

        nptest.assert_array_equal(lam_vol1,lam_vol3)
        self.compare_to_mean(lam_vol1)
        self.compare_to_vol_linear(lam_vol1)

    def compare_volume_ae(result_prob, result_mc):
        """

        Make sure that the voronoi cell volumes from 
        :meth:`~bet.calculateP.calculateP.prob`,
        :meth:`~bet.calculateP.calculateP.prob_mc` matches when the samples are
        i.i.d and ``lambda_emulate == samples``.

        .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.

        """
        # Calculate prob 
        (P1, lam_vol1, lem1, io_ptr1, emulate_ptr1) = result_prob
        (P3, lam_vol3, lem3, io_ptr3, emulate_ptr3) = result_mc

        nptest.assert_array_equal(lam_vol1, lam_vol3)
        self.compare_to_mean_ae(lam_vol1)
        self.compare_to_vol_linear_ae(lam_vol1)

    def compare_lambda_emulate(result_prob, result_rg, result_iid):
        """
        Compare results when lambda_emulate != samples
            
            * lambda_emulate is i.i.d. or on a regular grid
       
       .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.
       
        """
        # Calculate prob (has no lambda_emulate)
        (P1, lam_vol1, lem1, io_ptr1, emulate_ptr1) = result_prob
        # Calculate prob_mc (has lambda_emulate), regular grid
        (P3, lam_vol3, lem3, io_ptr3, emulate_ptr3) = result_rg
        # Calculate prob_mc (has lambda_emulate), iid samples
        (P4, lam_vol4, lem4, io_ptr4, emulate_ptr4) = result_iid

        # Compare results
        nptest.assert_array_almost_equal_nulp(P1,P3)
        nptest.assert_array_almost_equal_nulp(lam_vol1, lam_vol3)
        
        nptest.assert_array_almost_equal_nulp(P4,P3)
        nptest.assert_array_almost_equal_nulp(lam_vol4, lam_vol3)
        
        nptest.assert_array_almost_equal_nulp(P1,P4)
        nptest.assert_array_almost_equal_nulp(lam_vol1, lam_vol4)

    def generate_results(self):
        """
        Generate the mix of results from
        :meth:~`bet.calculateP.calculateP.prob_emulated`,
        :meth:~`bet.calculateP.calculateP.prob`, and
        :meth:~`bet.calculateP.calculateP.prob_mc` to be used in test
        subclasses
        """

        # RESULTS WHERE SAMPLES = LAMBDA_EMULATE
        # samples are on a regular grid
        # result_wtree, result_wsamples, result_emulated_rg
        self.result_emulated_rg = calcP.prob_emulated(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.r_samples, self.d_Tree)
        self.result_wtree = result_emulated_rg
        self.result_wsamples = result_emulated_rg
        self.result_wotree = calcP.prob_emulated(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.r_samples)
        self.result_wosamples = calcP.prob_emulated(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain)

        self.result_prob_rg = calcP.prob(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.d_Tree)
        self.result_mc_rg = calcP.prob_mc(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.r_samples, self.d_Tree)

        # samples are iid
        self.result_emulated_iid = calcP.prob_emulated(self.u_samples, self.u_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.u_samples, self.d_Tree)
        self.result_prob_iid = calcP.prob(self.u_samples, self.u_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.d_Tree)
        self.result_mc_iid = calcP.prob_mc(self.u_samples, self.u_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.u_samples, self.d_Tree)

        # RESULTS WHERE SAMPLES != LAMBDA_EMULATE
        # result_emu_samples_emulatedsamples
        self.result_emu_rg_rg = calcP.prob_mc(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.r_lambda_emulate, self.d_Tree)
        self.result_emu_rg_iid = calcP.prob_mc(self.r_samples, self.r_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.u_lambda_emulate, self.d_Tree)
        self.result_emu_iid_rg = calcP.prob_mc(self.u_samples, self.u_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.r_lambda_emulate, self.d_Tree)
        self.result_emu_iid_iid = calcP.prob_mc(self.u_samples, self.u_data,
                self.rho_D_M, self.d_distr_samples, self.lam_domain,
                self.u_lambda_emulate, self.d_Tree)


class LinearModel(CompareProb):
    """
    Tests ``prob*`` methods in :mod:`bet.calculateP.calculateP` with a linear
    model.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create inputs for ``prob*`` methods. This should run only once per
        grouping of tests from this class. But, for isolated tests it doesn't
        make any sense to do it this way. Since many of these tests are
        comparision tests it should be fine to structure things in this manner
        rather using inheritance everywhere.
        """
        # This might not be the right invocation.
        super(TestLinearModel, self).setUpClass(TestLinearModel)
        # linear model
        self.rl_data = np.dot(r_samples, data_domain)
        self.ul_data = np.dot(u_samples, data_domain)

class NonLinearModel(CompareProb):
    """
    Tests ``prob*`` methods in :mod:`bet.calculateP.calculateP` with a linear
    model.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create inputs for ``prob*`` methods. This should run only once per
        grouping of tests from this class. But, for isolated tests it doesn't
        make any sense to do it this way. Since many of these tests are
        comparision tests it should be fine to structure things in this manner
        rather using inheritance everywhere.
        """
        # This might not be the right invocation.
        super(NonLinearModel, self).setUpClass(NonLinearModel)
        # non-linear model
        def nonlinear_model(l_data):
            n_data = l_data
            n_data[:,0] = l_data[:,0] + l_data[:,2]
            n_data[:,1] = np.square(l_data[:,1])
            n_data[:,2] = l_data[:,0] - n_data[:,1]
            return n_data
        self.rn_data = nonlinear_model(rl_data)
        self.un_data = nonlinear_model(ul_data)

class TestProbUnifLinear(LinearModel, unittest.TestCase):
    """
    Tests ``prob*`` methods in :mod:`bet.calculateP.calculateP`.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create inputs for ``prob*`` methods. This should run only once per
        grouping of tests from this class. But, for isolated tests it doesn't
        make any sense to do it this way. Since many of these tests are
        comparision tests it should be fine to structure things in this manner
        rather using inheritance everywhere.
        """
        # This might not be the right invocation.
        super(TestProbUnifLinear, self).setUpClass(TestProbUnifLinear)
    
        # Create rho_D_M associated inputs (rho_D_M, d_distr_samples, d_tree)
        # UNIFORM
        self.d_distr_dsamples = np.mean(self.data_domain, 1)
        self.rho_D_M = 1
        self.d_Tree = spatial.KDTree(self.d_distr_samples)

        # Generate results
        super(TestProbUnifLinear, self).generate_results(self)

    def compare_to_unif_linear(result_P):
        """
        Compare P from the algorithm to an analytic solution where $P_\mathcal{D}$ is uniform
        over $\mathbf{D}$ (linear model only).

        :param result_P: P from any of the methods in
            :mod:`~bet.calculateP.calculateP`
        :type result_P: :class:`numpy.ndarray`

        """
        nptest.assert_array_equal(result_P,
                np.ones(result_P.shape)/float(len(result_P)))
        
    def compare_to_unif_linear_ae(result_P):
        """
        Compare P from the algorithm to an analytic solution where $P_\mathcal{D}$ is uniform
        over $\mathbf{D}$ (linear model only).

        :param result_P: P from any of the methods in
            :mod:`~bet.calculateP.calculateP`
        :type result_P: :class:`numpy.ndarray`

        """
        nptest.assert_array_almost_equal_nulp(result_P,
                np.ones(result_P.shape)/float(len(result_P)))

    def test_prob_dtree(self):
        """

        Make sure the output from
        :meth:`bet.calcuateP.calculateP.prob_emulated` matches with and without
        option arguments.

        """  
        compare_prob_dtree(self.result_wtree, self.result_wotree)

    def test_prob_emulate(self):
        """

        Make sure the output from
        :meth:`bet.calcuateP.calculateP.prob_emulated` matches with and without
        ``lambda_emulate`` when ``lambda_emulate == samples``.

        """
        compare_prob_emulate(self.result_wsamples, self.result_wosamples)

    def test_prob_rg(self):
        """

        Make sure that the output from 
        :meth:`~bet.calculateP.calculateP.prob_emulated`,
        :meth:`~bet.calculateP.calculateP.prob`,
        :meth:`~bet.calculateP.calculateP.prob_mc` matches when the samples are
        all on a regular grid when ``lambda_emulate == samples``.

        .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.

        """
        compare_prob(self.result_emulated_rg, self.result_prob_rg,self.result_mc_rg)
        compare_volume_rg(self.results_prob_rg, self.result_mc_rg)

    def test_prob_iid(self):
        """

        Make sure that the output from 
        :meth:`~bet.calculateP.calculateP.prob_emulated`,
        :meth:`~bet.calculateP.calculateP.prob`,
        :meth:`~bet.calculateP.calculateP.prob_mc` matches when the samples are
        i.i.d. with respect to the Lebesgue measure when ``lambda_emulate == samples``.

        .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.

        """
        compare_prob(self.result_emulated_iid, self.result_prob_iid,
                self.result_mc_iid)
        compare_volume_ae(self.results_prob_iid, self.result_mc_iid)
 
    def test_l_emulate_rg(self):
        """
        Compare results when lambda_emulate != samples
            
            * samples are on a regular grid
            * lambda_emulate is i.i.d. or on a regular grid
       
       .. note::
            This method also needs to include
            :meth:`~bet.calculateP.calculateP.prob_qhull` if and only if the
            user has the Python `pyhull <http://pythonhosted.org/pyhull>`_
            package installed.
       
        """
        compare_lambda_emulate(self.result_prob_rg, self.result_emu_rg_rg,
                self.result_emu_rg_iid)
        compare_volume_ae(
        compare_volume_ae(
        compare_volume_ae(

        # Calculate prob (has no lambda_emulate)
        (P1, lam_vol1, lem1, io_ptr1, emulate_ptr1) = calc.prob(self.u_samples,
                self.ul_data, self.u_rho, self.u_dsamples, self.lam_domain,
                self.ud_tree)

        # Calculate prob_mc (has lambda_emulate)
        (P3, lam_vol3, lem3, io_ptr3, emulate_ptr3) = calc.prob_mc(self.u_samples,
                self.ul_data, self.u_rho, self.u_dsamples, self.lam_domain,
                self.u_lambda_emulate, self.ud_tree)

        # Compare to mean
        self.compare_to_mean_ae(P3)
        self.compare_to_mean_ae(lam_vol3)

        (P4, lam_vol4, lem4, io_ptr4, emulate_ptr4) = calc.prob_mc(self.u_samples,
                self.ul_data, self.u_rho, self.u_dsamples, self.lam_domain,
                self.u_lambda_emulate, self.ud_tree)

        # Compare to mean
        self.compare_to_mean_ae(P4)
        self.compare_to_mean_ae(lam_vol4)

        # Compare results
        nptest.assert_array_almost_equal_nulp(P1,P3)
        nptest.assert_array_almost_equal_nulp(lam_vol1,lam_vol3)
        
        nptest.assert_array_almost_equal_nulp(P4,P3)
        nptest.assert_array_almost_equal_nulp(lam_vol4,lam_vol3)
        
        nptest.assert_array_almost_equal_nulp(P1,P4)
        nptest.assert_array_almost_equal_nulp(lam_vol1,lam_vol4
                )

    def test_compare_to_analytic_solution(self):
        pass
