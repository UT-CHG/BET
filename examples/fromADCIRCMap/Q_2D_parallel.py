import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import numpy as np
import scipy.io as sio
import bet.util as util
from bet.Comm import rank

# Import "Truth"
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q_ref = mdat['Q_true']

# Import Data
samples = mdat['points'].transpose()
lam_domain = np.array([[0.07, .15], [0.1, 0.2]])

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
    
    if rank == 0:
        print "Finished emulating lambda samples"
        mdict = dict()
        mdict['rho_D_M'] = rho_D_M
        mdict['d_distr_samples'] = d_distr_samples 
        mdict['num_l_emulate'] = num_l_emulate

    # Calculate P on lambda emulate
    (P0, lem0, io_ptr0, emulate_ptr0) = calcP.prob_emulated(samples, data,
            rho_D_M, d_distr_samples, lambda_emulate, d_Tree)
    if rank == 0:
        print "Calculating prob_emulated"
        mdict['P0'] = P0
        mdict['lem0'] = lem0
        mdict['io_ptr0'] = io_ptr0
        mdict['emulate_ptr0'] = emulate_ptr0

    # Calclate P on the actual samples with assumption that voronoi cells have
    # equal size
    (P1, lam_vol1, io_ptr1) = calcP.prob(samples, data,
            rho_D_M, d_distr_samples, d_Tree)
    if rank == 0:
        print "Calculating prob"
        mdict['P1'] = P1
        mdict['lam_vol1'] = lam_vol1
        mdict['lem1'] = samples
        mdict['io_ptr1'] = io_ptr1

    # Calculate P on the actual samples estimating voronoi cell volume with MC
    # integration
    (P3, lam_vol3, lambda_emulate3, io_ptr3, emulate_ptr3) = calcP.prob_mc(samples,
            data, rho_D_M, d_distr_samples, lambda_emulate, d_Tree)
    if rank == 0:
        print "Calculating prob_mc"
        mdict['P3'] = P3
        mdict['lam_vol3'] = lam_vol3
        mdict['io_ptr3'] = io_ptr3
        mdict['emulate_ptr3'] = emulate_ptr3
        # Export P and compare to MATLAB solution visually
        sio.savemat(filename, mdict, do_compression=True)

# Post-process and save P and emulated points
ref_nums = [6, 11, 15] # 7, 12, 16
stations = [1, 4, 5] # 2, 5, 6

ref_nums, stations = np.meshgrid(ref_nums, stations)
ref_nums = ref_nums.ravel()
stations = stations.ravel()

for tnum, stat in zip(ref_nums, stations):
    postprocess([0, stat], tnum)

