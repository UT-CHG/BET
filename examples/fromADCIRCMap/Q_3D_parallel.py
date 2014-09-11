import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import bet.visualize.plotP as pp
import numpy as np
import scipy.io as sio
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Import "Truth"
mdat = sio.loadmat('Q_3D')
Q = mdat['Q']
Q_true = mdat['Q_true']

# Import Data
samples = mdat['points'].transpose()
lam_domain = np.array([[-900, 1200], [0.07, .15], [0.1, 0.2]])

print "Finished loading data"

def postprocess(station_nums, true_num):
    
    filename = 'P_q'+str(station_nums[0]+1)+'_q'+str(station_nums[1]+1)
    if len(station_nums) == 3:
        filename += '_q'+str(station_nums[2]+1)
    filename += '_truth_'+str(true_num+1)

    data = Q[:, station_nums]
    q_true = Q_true[true_num, station_nums]

    # Create Simple function approximation
    # Save points used to parition D for simple function approximation and the
    # approximation itself (this can be used to make close comparisions...)
    (rho_D_M, d_distr_samples, d_Tree) = sfun.uniform_hyperrectangle(data,
            q_true, bin_ratio=0.15,
            center_pts_per_edge=np.ones((data.shape[1],)))

    num_l_emulate = 1e6
    lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
    print "Finished emulating lambda samples"

    # Calculate P on the actual samples estimating voronoi cell volume with MC
    # integration
    (P3, lam_vol3, lambda_emulate3, io_ptr3, emulate_ptr3) = calcP.prob_mc(samples,
            data, rho_D_M, d_distr_samples, lam_domain, lambda_emulate, d_Tree)
    print "Calculating prob_mc"
    mdict = dict()
    mdict['rho_D_M'] = rho_D_M
    mdict['d_distr_samples'] = d_distr_samples 
    mdict['lambda_emulate'] = pp.get_global_values(lambda_emulate)   
    mdict['num_l_emulate'] = mdict['lambda_emulate'].shape[1]
    mdict['P3'] = pp.get_global_values(P3)
    mdict['lam_vol3'] = pp.get_global_values(lam_vol3)
    mdict['io_ptr3'] = pp.get_global_values(io_ptr3)
    mdict['emulate_ptr3'] = emulate_ptr3
        
    if rank == 0:
        # Export P and compare to MATLAB solution visually
        sio.savemat(filename, mdict, do_compression=True)

# Post-process and save P and emulated points
true_num = 14

# q1, q5, q2 true 15
station_nums = [0, 4, 1] # 1, 5, 2
postprocess(station_nums, true_num)

# q1, q5 true 15
station_nums = [0, 4] # 1, 5
postprocess(station_nums, true_num)

# q1, q5, q12 true 16
#true_num = 15
station_nums = [0, 4, 11] # 1, 5, 12
postprocess(station_nums, true_num)


station_nums = [0, 8, 6] # 1, 5, 12
postprocess(station_nums, true_num)


station_nums = [0, 8, 11] # 1, 5, 12
postprocess(station_nums, true_num)


