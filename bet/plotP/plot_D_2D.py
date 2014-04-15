
import matplotlib.tri as tri
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
points = mdat['points']
points_true = mdat['points_true']
Q_true = mdat['Q_true']
triangulation = tri.Triangulation(points[0,:],points[1,:])
triangles = triangulation.triangles

figname = dict()
figname[4] = '2to2_'
figname[1] = '2to21_'
figname[5] = '2to22_'

img_folder = 'paper_figs_v41/'
for i in xrange(1,Q.shape[-1]):
    plt.tricontourf(Q[:,0],Q[:,i], np.zeros((Q.shape[0],)), triangles = triangles,
            colors = 'grey')
    #plt.gca().set_aspect('equal')
    plt.autoscale(tight=True)
    plt.xlabel(r'$q_1$')
    plt.ylabel(r'$q_{'+str(i+1)+r'}$')
    plt.savefig(img_folder+'scatter_q1_q'+str(i+1)+'_cs.eps', bbox_inches='tight',
            transparent=True, pad_inches=0)
    if i in [4, 1, 5]:
        plt.scatter(Q_true[6,0],Q_true[6,i], s = 60, c='r', marker = '^')
        plt.scatter(Q_true[11,0],Q_true[11,i], s = 60, c='g', marker = 's') 
        plt.scatter(Q_true[15,0],Q_true[15,i], s = 60, c='b', marker = 'o')
        plt.savefig(img_folder+figname[i]+'scatter_Q_cs.eps', bbox_inches='tight',
            transparent=True, pad_inches=0)
    plt.close()

