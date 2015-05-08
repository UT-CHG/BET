import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import numpy as np
import scipy.io as sio

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
    mdict = dict()
    mdict['rho_D_M'] = rho_D_M
    mdict['d_distr_samples'] = d_distr_samples

    # Calclate P on the actual samples with assumption that voronoi cells have
    # equal size
    (P1, lam_vol1, io_ptr1, emulate_ptr1) = calcP.prob(samples, data,
            rho_D_M, d_distr_samples, d_Tree)
    print "Calculating prob"
    mdict['P1'] = P1
    mdict['lam_vol1'] = lam_vol1
    mdict['lem1'] = samples
    mdict['io_ptr1'] = io_ptr1
    mdict['emulate_ptr1'] = emulate_ptr1

    # Export P and compare to MATLAB solution visually
    sio.savemat(filename, mdict, do_compression=True)

# Post-process and save P and emulated points
ref_num = 14

# q1, q5, q2 ref 15
station_nums = [0, 4, 1] # 1, 5, 2
postprocess(station_nums, ref_num)

# q1, q5 ref 15
station_nums = [0, 4] # 1, 5
postprocess(station_nums, ref_num)

# q1, q5, q12 ref 16
#ref_num = 15
station_nums = [0, 4, 11] # 1, 5, 12
postprocess(station_nums, ref_num)


station_nums = [0, 8, 6] # 1, 5, 12
postprocess(station_nums, ref_num)


station_nums = [0, 8, 11] # 1, 5, 12
postprocess(station_nums, ref_num)


