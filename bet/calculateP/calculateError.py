from bet.Comm import comm, MPI 
import numpy as np
import math
import logging
import scipy.spatial as spatial
import bet.util as util
import bet.calculateP.calculateP as calculateP
import bet.postProcess.postTools as postTools
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
    """
    from scipy.spatial import Delaunay
    from collections import defaultdict
    import itertools
    import numpy.linalg as nlinalg

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
    if disc.get_io_ptr() is None:
        disc.set_io_ptr()
    tri = Delaunay(disc._input_sample_set._values)
    neiList=defaultdict(set)
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i].add(disc._io_ptr[j])
            neiList[j].add(disc._io_ptr[i])
    for i in range(num):
        neiList[i] = list(set(neiList[i]))
    return neiList

def boundary_sets(disc, nei_list):
    from collections import defaultdict

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
    if disc.get_io_ptr() is None:
        disc.set_io_ptr()
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
    def __init__(self, disc, exact=True):
        if not isinstance(disc, samp.discretization):
            msg = "The argument must be of type bet.sample.discretization."
            raise wrong_argument_type(msg)

        self.disc = disc 
        self.num = self.disc.check_nums()
        if exact:
            nei_list = cell_connectivity_exact(self.disc)
        else:
            msg = "sampling_error only handles exact connectivity for now."
            raise type_not_yet_written(msg)
        
        (self.B_N, self.C_N) = boundary_sets(self.disc, nei_list)
        
    def calculate_for_contour_events(self):
        up_list = []
        low_list = []
        if self.disc._input_sample_set._volumes is None:
            if self.disc._emulated_input_sample_set is not None:
                logging.warning("Using emulated points to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_emulated(self.disc._emulated_input_sample_set)
            else:
                logging.warning("Making MC assumption to estimate volumes.")
                self.disc._input_sample_set.estimate_volume_mc()
                
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

    def calculate_for_marked_sample_set(self, s_set, marker, emulated_set=None):
        if emulated_set is not None:
            disc = self.disc.copy()
            disc.set_emulated_input_sample_set(emulated_set)
            disc.set_emulated_ii_ptr()
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = emulated_set)
            disc_new.set_emulated_ii_ptr()
        elif self.disc._emulated_input_sample_set is not None:
            disc = self.disc
            if disc._emulated_ii_ptr is None:
                disc.set_emulated_ii_ptr()
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = self.disc._emulated_input_sample_set)
            
            disc_new.set_emulated_ii_ptr()
        else:
            disc = self.disc.copy()
            disc.set_emulated_input_sample_set(disc._input_sample_set._values)
            disc.set_emulated_ii_ptr()
            
            disc_new = samp.discretization(input_sample_set = s_set,
                                           output_sample_set = s_set,
                                           emulated_input_sample_set = self.disc._input_sample_set)
            disc_new.set_emulated_ii_ptr()
            
        in_A = marker[disc_new._emulated_ii_ptr_local]
        
        upper_bound = 0.0
        lower_bound = 0.0
    
        ops_num = self.disc._output_probability_set.check_num()
        for i in range(ops_num):
            if self.disc._output_probability_set._probabilities[i] > 0.0:
                indices = np.equal(disc._io_ptr_local, i)
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