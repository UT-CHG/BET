import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import numpy as np
import scipy.io as sio
from bet import util

from bet.Comm import comm, rank

# Import "Truth"
mdat = sio.loadmat('Q_3D')
Q = mdat['Q']
Q_ref = mdat['Q_true']

# Import Data
samples = mdat['points'].transpose()
lam_domain = np.array([[-900, 1200], [0.07, .15], [0.1, 0.2]])

print "Finished loading data"

def postprocess(station_nums, ref_num):
    
    filename = 'P_q'+str(station_nums[0]+1)+'_q'+str(station_nums[1]+1)
    if len(station_nums) == 3:
        filename += '_q'+str(station_nums[2]+1)
    filename += '_ref_'+str(ref_num+1)

    data = Q[:, station_nums]
    q_ref = Q_ref[ref_num, station_nums]

    # Create Simple function approximation
    # Save points used to parition D for simple function approximation and the
    # approximation itself (this can be used to make close comparisions...)
    (rho_D_M, d_distr_samples, d_Tree) = sfun.uniform_hyperrectangle(data,
            q_ref, bin_ratio=0.15,
            center_pts_per_edge=np.ones((data.shape[1],)))

    num_l_emulate = 1e6
    lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
    print "Finished emulating lambda samples"

    # Calculate P on the actual samples estimating voronoi cell volume with MC
    # integration
    print "Calculating prob_mc"
    (P3, lam_vol3, lambda_emulate3, io_ptr3, emulate_ptr3) = calcP.prob_mc(samples,
            data, rho_D_M, d_distr_samples, lambda_emulate, d_Tree)

    if rank == 0:
        mdict = dict()
        mdict['rho_D_M'] = rho_D_M
        mdict['d_distr_samples'] = d_distr_samples 
        mdict['lambda_emulate'] = util.get_global_values(lambda_emulate)   
        mdict['num_l_emulate'] = mdict['lambda_emulate'].shape[1]
        mdict['P3'] = util.get_global_values(P3)
        mdict['lam_vol3'] = lam_vol3
        mdict['io_ptr3'] = io_ptr3
        mdict['emulate_ptr3'] = emulate_ptr3
        print "Exporting P"    
        # Export P 
        sio.savemat(filename, mdict, do_compression=True)

# Post-process and save P and emulated points
ref_num = 14

# q1, q5, q2 ref 15
station_nums = [0, 4, 1] # 1, 5, 2
postprocess(station_nums, ref_num)

"""
# q1, q5 ref 15
station_nums = [0, 4] # 1, 5
postprocess(station_nums, ref_num)

# q1, q5, q12 ref 16
station_nums = [0, 4, 11] # 1, 5, 12
postprocess(station_nums, ref_num)


station_nums = [0, 8, 6] # 1, 5, 12
postprocess(station_nums, ref_num)


station_nums = [0, 8, 11] # 1, 5, 12
postprocess(station_nums, ref_num)
"""

