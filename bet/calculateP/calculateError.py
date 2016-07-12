# Copyright (C) 2014-2016 The BET Development Team

r""" 
This module provides methods for calulating error estimates of 
the probability measure for calculate probability measures.

* :mod:`~bet.calculateErrors.cell_connectivity_exact` calculates 
the connectivity of cells.

"""

from bet.Comm import comm, MPI 
import numpy as np
import math
import logging
import scipy.spatial as spatial
import bet.util as util
import bet.calculateP.calculateP as calculateP
import bet.sample as samp

class wrong_argument_type(Exception):
    """
    Exception for when the argument is not one of the acceptible
    types.
    """
class type_not_yet_written(Exception):
    """
    Exception for when the method is not yet written.
    """

def cell_connectivity_exact(disc):
    """
    
    Calculates the the neighboring Voronoi cells for each cell.

    :param disc: An object containing the discretization information.
    :type disc: class:`bet.sample.discretization`

    :rtype: list
    :returns: list of lists of neighboring cells

    """
    from scipy.spatial import Delaunay
    from collections import defaultdict
    import itertools
    import numpy.linalg as nlinalg

    # Check inputs
    if not isinstance(disc, samp.discretization):
        msg = "The argument must be of type bet.sample.discretization."
        raise wrong_argument_type(msg)

    if not isinstance(disc._input_sample_set, samp.voronoi_sample_set):
        msg = "disc._input_sample_set must be of type bet.sample.voronoi_sample_set defined with the 2-norm"
        raise wrong_argument_type(msg)
    elif disc._input_sample_set._p_norm != 2.0:
        msg = "disc._input_sample_set must be of type bet.sample.voronoi_sample_set defined with the 2-norm"
        raise wrong_argument_type(msg)

    num = disc.check_nums()
    # Set up necessary pointers
    if disc.get_io_ptr() is None:
        disc.set_io_ptr()
    # Form Delaunay triangulation
    tri = Delaunay(disc._input_sample_set._values)

    # Find neighbors
    neiList=defaultdict(set)
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i].add(disc._io_ptr[j])
            neiList[j].add(disc._io_ptr[i])
    # Get rid of redundant entries
    for i in range(num):
        neiList[i] = list(set(neiList[i]))

    return neiList

def boundary_sets(disc, nei_list):
    """
    
    Calculates the the neighboring Voronoi cells for each cell.

    :param disc: An object containing the discretization information.
    :type disc: class:`bet.sample.discretization`
    :param nei_list: list of lists defining neighboring cells.
    :type nei_list: list

    :rtype: tuple
    :returns: (B_N, C_N) as defined in Butler et al. 2015.

    """
    from collections import defaultdict

    # Check inputs
    if not isinstance(disc, samp.discretization):
        msg = "The argument must be of type bet.sample.discretization."
        raise wrong_argument_type(msg)

    if not isinstance(disc._input_sample_set, samp.voronoi_sample_set):
        msg = "disc._input_sample_set must be of type bet.sample.voronoi_sample_set defined with the 2-norm"
        raise wrong_argument_type(msg)
    elif disc._input_sample_set._p_norm != 2.0:

        msg = "disc._input_sample_set must be of type bet.sample.voronoi_sample_set defined with the 2-norm"
        raise wrong_argument_type(msg)

    num = disc.check_nums()
    
    # Form necessary pointers
    if disc.get_io_ptr() is None:
        disc.set_io_ptr()
    # Define strictly interior and boundary cells for each contour event
    B_N = defaultdict(list)
    C_N = defaultdict(list)
    for i in range(num):
        contour_event = disc._io_ptr[i]
        if nei_list[i] == [contour_event]:
            B_N[contour_event].append(i)
        for j in nei_list[i]:
            C_N[j].append(i)
    
    return (B_N, C_N)

class sampling_error(object):
    """
    A class for calculating the error due to sampling for a discretization.
    """
    def __init__(self, disc, exact=True):
        """
          
        Set things up for a given discretization
          
        :param disc: An object containing the discretization information.
        :type disc: class:`bet.sample.discretization`
        :param exact: Whether or not to use exact connectivity
        :type exact: bool
        
        """
        # Check inputs
        if not isinstance(disc, samp.discretization):
            msg = "The argument must be of type bet.sample.discretization."
            raise wrong_argument_type(msg)

        self.disc = disc 
        self.num = self.disc.check_nums()

        # Set up neighbor list and B_N and C_N
        if exact:
            nei_list = cell_connectivity_exact(self.disc)
        else:
            msg = "sampling_error only handles exact connectivity for now."
            raise type_not_yet_written(msg)
        
        (self.B_N, self.C_N) = boundary_sets(self.disc, nei_list)
        
    def calculate_for_contour_events(self):
        """
        Calculate the sampling error bounds for each contour event.

        :rtype tuple
        :returns: (up_list, low_list) where up_list is a list of
        the upper bounds for each contour event and low_list is a list
        of the lower bounds.

        """
        up_list = []
        low_list = []
        # Check for and possibly calculate volumes
        if self.disc._input_sample_set._volumes is None:
            if self.disc._emulated_input_sample_set is not None:
                logging.warning("Using emulated points to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_emulated(self.disc._emulated_input_sample_set)
            else:
                logging.warning("Making MC assumption to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_mc()

        # Loop over contour events and calculate error bounds
        ops_num = self.disc._output_probability_set.check_num()
        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:
                lam_vol = np.zeros((self.num,))
                indices = np.equal(self.disc._io_ptr,i)
                lam_vol[indices] = self.disc._input_sample_set._volumes[indices]
                if i in self.B_N:
                    val1 = np.sum(self.disc._input_sample_set._volumes[self.B_N[i]])
                    val2 = np.sum(lam_vol[self.B_N[i]])
                
                    val3 = np.sum(self.disc._input_sample_set._volumes[self.C_N[i]])
                    val4 = np.sum(lam_vol[self.C_N[i]])
                
                    term1 = val2/val3 - 1.0
                    term2 = val4/val1 - 1.0
                    up_list.append(self.disc._output_probability_set._probabilities[i]*max(term1,term2))
                    low_list.append(self.disc._output_probability_set._probabilities[i]*min(term1,term2))
                else:
                    up_list.append(float('nan'))
                    low_list.append(float('nan'))
                
            else:
                up_list.append(0.0)
                low_list.append(0.0)

        return (up_list, low_list)

    def calculate_for_sample_set_region(self, s_set, 
                                     region, emulated_set=None):
        """
        Calculate the sampling error bounds for a region of the input space
        defined by a sample set object.

        :param s_set: sample set for which to calculate error
        :type s_set: :class:`bet.sample.sample_set_base`
        :param region: region of s_set for which to calculate error
        :type region: int
        :param emulated_set: sample set for volume emulation
        :type emulated_set: :class:`bet.sample_set_base`

        :rtype tuple
        :returns: (upper_bound, lower_bound) the upper and lower bounds
        for the error.

        """
        # Set up marker
        self.disc._input_sample_set.local_to_global()
        if s_set._region is None:
            msg = "regions must be defined for the sample set."
            raise wrong_argument_type(msg)
        marker = np.equal(s_set._region, region)
        if not np.any(marker):
            msg = "The given region does not exist."
            raise wrong_argument_type(msg)
            
            
        # Set up discretizations
        if emulated_set is not None:
            disc = self.disc.copy()
            disc.set_emulated_input_sample_set(emulated_set)
            disc.set_emulated_ii_ptr(globalize=False)
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = emulated_set)
            disc_new.set_emulated_ii_ptr(globalize=False)
        elif self.disc._emulated_input_sample_set is not None:
            logging.warning("Using emulated_input_sample_set for volume emulation")
            disc = self.disc
            if disc._emulated_ii_ptr is None:
                disc.set_emulated_ii_ptr(globalize=False)
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = self.disc._emulated_input_sample_set)
            
            disc_new.set_emulated_ii_ptr(globalize=False)
        else:
            logging.warning("Using input_sample_set for volume emulation")
            disc = self.disc.copy()
            disc.set_emulated_input_sample_set(disc._input_sample_set._values)
            disc.set_emulated_ii_ptr(globalize=False)
            
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = self.disc._input_sample_set)
            disc_new.set_emulated_ii_ptr(globalize=False)
        
        # Emulated points in the the region
        in_A = marker[disc_new._emulated_ii_ptr_local]
        
        upper_bound = 0.0
        lower_bound = 0.0
        # Loop over contour intervals and add error contributions
        ops_num = self.disc._output_probability_set.check_num()
        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:
                indices = np.equal(disc._io_ptr, i)
                in_Ai = indices[disc._emulated_ii_ptr_local]
                
                sum1 = np.sum(np.logical_and(in_A, in_Ai))
                sum2 = np.sum(in_Ai)
                sum1 = comm.allreduce(sum1, op=MPI.SUM)
                sum2 = comm.allreduce(sum2, op=MPI.SUM)
                E = float(sum1)/float(sum2)

                in_B_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.B_N[i]:
                    in_B_N = np.logical_or(np.equal(disc._emulated_ii_ptr_local,j),in_B_N)

                in_C_N = np.zeros(in_A.shape, dtype=np.bool)
                for j in self.C_N[i]:
                    in_C_N =  np.logical_or(np.equal(disc._emulated_ii_ptr_local,j), in_C_N)

                sum3 = np.sum(np.logical_and(in_A,in_B_N))
                sum4 = np.sum(in_C_N)
                sum3 = comm.allreduce(sum3, op=MPI.SUM)
                sum4 = comm.allreduce(sum4, op=MPI.SUM)
                term1 = float(sum3)/float(sum4) - E
                sum5 = np.sum(np.logical_and(in_A,in_C_N))
                sum6 = np.sum(in_B_N)
                sum5 = comm.allreduce(sum5, op=MPI.SUM)
                sum6 = comm.allreduce(sum6, op=MPI.SUM)
                term2 = float(sum5)/float(sum6) - E

                upper_bound += self.disc._output_probability_set._probabilities[i]*max(term1,term2)
                lower_bound += self.disc._output_probability_set._probabilities[i]*min(term1,term2)
        return (upper_bound, lower_bound)
                                       

class model_error(object):
    """
    A class for calculating the error due to numerical error
    for a discretization.
    """
    def __init__(self, disc):
        """
          
        Set things up for a given discretization
          
        :param disc: An object containing the discretization information.
        :type disc: class:`bet.sample.discretization`
        
        """
        # Check inputs
        if not isinstance(disc, samp.discretization):
            msg = "The argument must be of type bet.sample.discretization."
            raise wrong_argument_type(msg)
        disc._output_sample_set.global_to_local()
        self.disc = disc 
        if self.disc._output_sample_set._error_estimates is None:
            if self.disc._output_sample_set._error_estimates_local is None:
                msg = "Error estimates for the output sample set are required."
                raise wrong_argument_type(msg)
        #self.disc._output_sample_set.global_to_local()

        self.num = self.disc.check_nums()
        if self.disc._io_ptr_local is None:
            self.disc.set_io_ptr()

        # Setup new discretization object adding error estimates
        self.disc_new = disc.copy()
        
        # Think about parallelism here
        #self.disc_new._output_sample_set._values_local = None
        #if self.disc._output_sample_set._error_estimates_local is None:
        #   self.disc._output_sample_set.global_to_local() 
        self.disc_new._output_sample_set._values_local += self.disc._output_sample_set._error_estimates_local
        self.disc_new.set_io_ptr(globalize=False) #True)
        self.disc_new._io_ptr = None
        

    def calculate_for_contour_events(self):
        """
        Calculate the numerical error for each contour event.

        :rtype list
        :returns: er_list, a list of the error estimates
        for each contour event.

        """
        # Calculate volumes if necessary
        if self.disc._input_sample_set._volumes is None:
            if self.disc._emulated_input_sample_set is not None:
                logging.warning("Using emulated points to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_emulated(self.disc._emulated_input_sample_set)
            else:
                logging.warning("Making MC assumption to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_mc()
        # Localize if necessary
        if self.disc._input_sample_set._volumes_local is None:
            self.disc._input_sample_set.global_to_local()

        # Loop over contour events and add contributions
        er_list = []
        ops_num = self.disc._output_probability_set.check_num()
        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:                
                ind1 = np.equal(self.disc._io_ptr_local, i)
                ind2 = np.equal(self.disc_new._io_ptr_local, i)
                JiA = np.sum(self.disc._input_sample_set._volumes_local[ind1])
                Ji = JiA
                JiAe = np.sum(self.disc._input_sample_set._volumes_local[np.logical_and(ind1,ind2)])
                Jie = np.sum(self.disc._input_sample_set._volumes_local[ind2])

                JiA = comm.allreduce(JiA, op=MPI.SUM)
                Ji = comm.allreduce(Ji, op=MPI.SUM)
                JiAe = comm.allreduce(JiAe, op=MPI.SUM)
                Jie = comm.allreduce(Jie, op=MPI.SUM)
                er_list.append(self.disc._output_probability_set._probabilities[i] * ((JiA*Jie - JiAe*Ji)/(Ji*Jie)))
            else:
                er_list.append(0.0)

        return er_list

    def calculate_for_sample_set_region(self, s_set, 
                                    region, emulated_set=None):
        """
        Calculate the numerical error estimate for a region of the input space
        defined by a sample set object.

        :param s_set: sample set for which to calculate error
        :type s_set: :class:`bet.sample.sample_set_base`
        :param region: region of s_set for which to calculate error
        :type region: int
        :param emulated_set: sample set for volume emulation
        :type emulated_sample_set: :class:`bet.sample_set_base`

        :rtype float
        :returns: er_est, the numerical error estimate for the region

        """
        # Set up marker
        if s_set._region is None:
            msg = "regions must be defined for the sample set."
            raise wrong_argument_type(msg)
        marker = np.equal(s_set._region, region)
        if not np.any(marker):
            msg = "The given region does not exist."
            raise wrong_argument_type(msg)

        # Setup discretizations
        if emulated_set is not None:
            self.disc._input_sample_set.local_to_global()
            self.disc.globalize_ptrs()
            self.disc_new.globalize_ptrs()

            disc = self.disc.copy()
            disc.set_emulated_input_sample_set(emulated_set)
            disc.set_emulated_ii_ptr(globalize=False)
        
            #disc_new = self.disc_new.copy()
            #disc_new.set_emulated_input_sample_set(emulated_set)
            #disc_new.set_emulated_ii_ptr(globalize=False)
            disc_new_set = samp.discretization(input_sample_set = s_set,
                                               output_sample_set = s_set,
                                               emulated_input_sample_set = emulated_set)
            disc_new_set.set_emulated_ii_ptr(globalize=False)
        elif self.disc._emulated_input_sample_set is not None:
            self.disc._input_sample_set.local_to_global()
            logging.warning("Using emulated_input_sample_set for volume emulation")
            self.disc.globalize_ptrs()
            self.disc_new.globalize_ptrs()
            disc = self.disc
            if disc._emulated_ii_ptr_local is None:
                disc.set_emulated_ii_ptr(globalize=False)
            #disc_new = self.disc_new.copy()
            disc_new.set_emulated_ii_ptr(globalize=False)
            disc_new_set = samp.discretization(input_sample_set = s_set,
                                               output_sample_set = s_set,
                                               emulated_input_sample_set = disc._emulated_input_sample_set)
            disc_new_set.set_emulated_ii_ptr(globalize=False)
        else:
            logging.warning("Using MC assumption for volumes.")
            return self.calculate_for_sample_set_region_mc(s_set, region)
            # # Next two lines not necessary for this case. should make new method
            # self.disc.globalize_ptrs()
            # self.disc_new.globalize_ptrs()
            # disc = self.disc.copy()
            # disc.set_emulated_input_sample_set(disc._input_sample_set._values)
            # disc.set_emulated_ii_ptr(globalize=False)
            # disc_new = self.disc_new.copy()
            # disc_new.set_emulated_input_sample_set(disc._input_sample_set._values)
            # disc_new.set_emulated_ii_ptr(globalize=False)
            # disc_new_set = samp.discretization(input_sample_set=s_set,
            #                                    output_sample_set=s_set,
            #                                    emulated_input_sample_set=disc._input_sample_set._values)
            # disc_new_set.set_emulated_ii_ptr(globalize=False)
        
        # Setup pointers
        ptr1 = disc._emulated_ii_ptr_local
        #ptr2 = disc_new._emulated_ii_ptr_local
        ptr3 = disc_new_set._emulated_ii_ptr_local
                
        # Check if in the region
        in_A = marker[ptr3]

        # Loop over contour events and add error contribution
        er_est = 0.0
        ops_num = self.disc._output_probability_set.check_num()

        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:
                indices1 = np.equal(self.disc._io_ptr ,i)
                in_Ai1 = indices1[ptr1]
                indices2 = np.equal(self.disc_new._io_ptr ,i)
                in_Ai2 = indices2[ptr1]
                JiA_local = float(np.sum(np.logical_and(in_A,in_Ai1)))
                JiA = comm.allreduce(JiA_local, op=MPI.SUM)
                Ji_local = float(np.sum(in_Ai1))
                Ji = comm.allreduce(Ji_local, op=MPI.SUM)
                JiAe_local = float(np.sum(np.logical_and(in_A,in_Ai2)))
                JiAe = comm.allreduce(JiAe_local, op=MPI.SUM)
                Jie_local = float(np.sum(in_Ai2))
                Jie = comm.allreduce(Jie_local, op=MPI.SUM)
                er_est += self.disc._output_probability_set._probabilities[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))
               
        return er_est

    def calculate_for_sample_set_region_mc(self, s_set, 
                                       region):
        """
        Calculate the numerical error estimate for a region of the input space
        defined by a sample set object, using the MC assumption.

        :param s_set: sample set for which to calculate error
        :type s_set: :class:`bet.sample.sample_set_base`
        :param region: region of s_set for which to calculate error
        :type region: int

        :rtype float
        :returns: er_est, the numerical error estimate for the region

        """
        # Set up marker
        if s_set._region is None:
            msg = "regions must be defined for the sample set."
            raise wrong_argument_type(msg)
        marker = np.equal(s_set._region, region)
        if not np.any(marker):
            msg = "The given region does not exist."
            raise wrong_argument_type(msg)

        disc_new_set = samp.discretization(input_sample_set=s_set,
                                           output_sample_set=s_set,
                                           emulated_input_sample_set=self.disc._input_sample_set)
        disc_new_set.set_emulated_ii_ptr(globalize=False)
        #ptr3 = disc_new_set._emulated_ii_ptr_local
        # Check if in the region
        in_A = marker[disc_new_set._emulated_ii_ptr_local]

        # Loop over contour events and add error contribution
        er_est = 0.0
        ops_num = self.disc._output_probability_set.check_num()
        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:
                #indices1 = np.equal(self.disc._io_ptr ,i)
                #in_Ai1 = indices1[ptr1]
                in_Ai1 = np.equal(self.disc._io_ptr_local, i)
                #indices2 = np.equal(self.disc_new._io_ptr ,i)
                #in_Ai2 = indices2[ptr1]
                in_Ai2 = np.equal(self.disc_new._io_ptr_local, i)
                JiA_local = float(np.sum(np.logical_and(in_A,in_Ai1)))
                JiA = comm.allreduce(JiA_local, op=MPI.SUM)
                Ji_local = float(np.sum(in_Ai1))
                Ji = comm.allreduce(Ji_local, op=MPI.SUM)
                JiAe_local = float(np.sum(np.logical_and(in_A,in_Ai2)))
                JiAe = comm.allreduce(JiAe_local, op=MPI.SUM)
                Jie_local = float(np.sum(in_Ai2))
                Jie = comm.allreduce(Jie_local, op=MPI.SUM)
                er_cont = self.disc._output_probability_set._probabilities[i]*((JiA*Jie - JiAe*Ji)/(Ji*Jie))
                er_est += er_cont
                error_cells1 = np.logical_and(np.logical_and(in_Ai1, np.logical_not(in_A)), np.logical_and(in_A2, in_A))
                error_cells2 = np.logical_and(np.logical_and(in_Ai2, np.logical_not(in_A)), np.logical_and(in_A1, in_A))
                #error_cells3 = np.logical_and(in_Ai1, np.logical_not(in_Ai2))
                #error_cells4 = np.logical_and(in_Ai2, np.logical_not(in_Ai1))
                error_cells3 = np.not_equal(in_Ai1, in_Ai2)
                error_cells = np.logical_or(error_cells1, error_cells2)
                error_cells = np.logical_or(error_cells, error_cells3)
                #error_cells = np.logical_or(error_cells, error_cells4)

                error_cells_num_local = float(np.sum(error_cells))
                error_cells_num = comm.allreduce(error_cells_num_local, op=MPI.SUM)
                self.disc._output_sample_set._error_id_local += er_cont/error_cells_num
               
        return er_est
