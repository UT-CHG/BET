
"""
This module approximates gradient information
    of each of the QoI maps.
"""
import numpy as np
import scipy.spatial as spatial
import sys
import scipy
from itertools import combinations

def calculate_gradients_rbf(samples, data, xeval, num_neighbors=None, RBF=None, ep=None):
    r"""

    TO DO: vectorize first for loop?

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim is the 
        dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is the number
        of QoI (i.e. the dimension of the data space :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param int num_neighbors: Number of nearest neighbors to use in gradient approximation.
        Default value is 30.
    :param string RBF: Choice of radial basis function.
        Default is Gaussian
    :param float ep: Choice of shape parameter for radial basis function.
        Default value is 1.0
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    if num_neighbors is None:
        num_neighbors = 30
    if ep is None:
        ep = 1.0
    if RBF is None:
       RBF = 'Gaussian'

    Lambda_dim = samples.shape[1]
    num_model_samples = samples.shape[0]
    Data_dim = data.shape[1]
    num_xeval = xeval.shape[0]

    rbf_tensor = np.zeros([num_xeval, num_model_samples, Lambda_dim])
    gradient_tensor = np.zeros([num_xeval, Data_dim, Lambda_dim])
    tree = spatial.KDTree(samples)
    
    for xe in range(num_xeval):
        [r, nearest] = tree.query(xeval[xe,:], k=num_neighbors)
        r = np.tile(r, (Lambda_dim,1))
        #nearest = np.tile(nearest, (Lambda_dim,1))


        diffVec = (xeval[xe,:] - samples[nearest,:]).transpose()
        distMat = scipy.spatial.distance_matrix(samples[nearest, :], samples[nearest, :])
        rbf_mat_values = np.linalg.solve(radial_basis_function(distMat, RBF), \
            radial_basis_function_dxi(r, diffVec, RBF, ep).transpose()).transpose()

        for ind in range(num_neighbors):
            rbf_tensor[xe, nearest[ind], :] = rbf_mat_values[:, ind].transpose()

    gradient_tensor = rbf_tensor.transpose(2,0,1).dot(data).transpose(1,2,0)
    
   
    return gradient_tensor

def calculate_gradients_cfd(samples, data, xeval, r):
    r"""

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.
    THIS METHOD IS DEPENDENT ON USING pick_cfd_points TO CHOOSE
    SAMPLES FOR THE CFD STENCIL AROUND EACH XEVAL

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim is the 
        dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is the number
        of QoI (i.e. the dimension of the data space :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Distance from center to place samples
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    gradient_vec = (data[:Lambda_dim*num_xeval] - data[Lambda_dim*num_xeval:])/(2*r)
    gradient_tensor = gradient_vec.reshape(Lambda_dim,1,num_xeval).transpose(2,1,0)

    return gradient_tensor

def calculate_gradients_ffd(samples, data, xeval, r):
    r"""

    Approximate gradient vectors at ``num_xeval, xeval.shape[0]`` points
    in the parameter space for each QoI map.
    THIS METHOD IS DEPENDENT ON USING pick_cfd_points TO CHOOSE
    SAMPLES FOR THE CFD STENCIL AROUND EACH XEVAL

    :param samples: Samples for which the model has been solved.
    :type samples: :class:`np.ndarray` of shape (num_samples, Ldim) where Ldim is the 
        dimension of the parameter space :math:`\Lambda`
    :param data: QoI values corresponding to each sample.
    :type data: :class:`np.ndarray` of shape (num_samples, Ddim) where Ddim is the number
        of QoI (i.e. the dimension of the data space :math:`\mathcal{D}`
    :param xeval: Points in :math:`\Lambda` at which to approximate gradient
        information.
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Distance from center to place samples
    :rtype: :class:`np.ndarray` of shape (num_samples, Ddim, Ldim)
    :returns: Tensor representation of the gradient vectors of each QoI map
        at each point in xeval

    """
    num_xeval = xeval.shape[0]
    Lambda_dim = samples.shape[1]
    num_qois = data.shape[1]
    gradient_tensor = np.zeros([num_xeval, num_qois, Lambda_dim])

    gradient_vec = (data[num_xeval:] - np.tile(data[0:num_xeval], [Lambda_dim,1]))/r
    gradient_tensor = gradient_vec.reshape(Lambda_dim,1,num_xeval).transpose(2,1,0)

    return gradient_tensor

def radial_basis_function(r, kernel=None, ep=None):
    """

    Evaluate a chosen radial basis function.  Allow for the 
    choice of several radial basis functions to use in
    the calculate_gradients_rbf.

    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion
    :param float ep: Shape parameter for the radial basis function
    :rtype: :class:`np.ndarray` of shape (r.shape)
    :returns: Radial basis function evaluated for each element of r

    """
    if ep is None:
        ep = 1.0

    if kernel is None or kernel is 'C4Matern':
        rbf = (1+(ep*r)+(ep*r)**2/3)*np.exp(-ep*r)
    elif kernel is 'Gaussian':
        rbf = np.exp(-(ep*r)**2)
    elif kernel is 'Multiquadric':
        rbf = (1+(ep*r)**2)**(0.5)
    elif kernel is 'InverseMultiquadric':
        rbf = 1/((1+(ep*r)**2)**(0.5))
    else:
        raise ValueError("The kernel chosen is not currently available.")

    return rbf

def radial_basis_function_dxi(r, xi, kernel=None, ep=None):
    """

    Evaluate a partial derivative of a chosen radial basis function.
    Allow for the choice of several radial basis functions to
    use in the calculate_gradients_rbf.

    :param r: Distances from the reference point
    :type r: :class:`np.ndarray`
    :param xi: Distances from the reference point in dimension i
    :type xi: :class:`np.ndarray`
    :param string kernel: Choice of radial basis funtion
    :param float ep: Shape parameter for the radial basis function
    :rtype: :class:`np.ndarray` of shape (r.shape)
    :returns: Radial basis function evaluated for each element of r

    """
    if ep is None:
        ep = 1.0
    
    if kernel is None or kernel is 'C4Matern':
        rbfdxi = -(ep**2*xi*np.exp(-ep*r)*(ep*r + 1))/3
    elif kernel is 'Gaussian':
        rbfdxi = -2*ep**2*xi*np.exp(-(ep*r)**2)
    elif kernel is 'Multiquadric':
        rbfdxi = (ep**2*xi)/((1+(ep*r)**2)**(0.5))
    elif kernel is 'InverseMultiquadric':
        rbfdxi = -(ep**2*xi)/((1+(ep*r)**2)**(1.5))
    else:
        raise ValueError("The kernel chosen is not currently available")

    return rbfdxi

def pick_close_points(num_close, xeval, r):
    """

    Pick num_close points in a box of length 2*r around a
    point in :math:`\Lambda`, do this for each point
    in xeval.

    :param int num_close: Number of points in each cluster
    :param xeval: Points in :math:`\Lambda` to cluster points around
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_xeval, Ldim)
    :returns: Clusters of samples near each point in xeval

    """

    Lambda_dim = xeval.shape[1]
    num_xeval = xeval.shape[0]
    a = xeval - r
    b = xeval + r

    samples_temp = np.dot((b-a), np.random.random([Lambda_dim,Lambda_dim*num_close])) + np.tile(a,[1,num_close])
    samples = samples_temp.reshape(num_xeval*num_close, Lambda_dim)

    return samples

def pick_cfd_points(xeval, r):
    """

    Pick 2*Lambda_dim points, for each xeval, for centered 
    finite diff grad approx.

    :param xeval: Points in :math:`\Lambda` to cluster points around
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_xeval, Ldim)
    :returns: Samples for centered finite difference stencil for 
        each point in xeval.


    """
    Lambda_dim = xeval.shape[1]
    num_xeval = xeval.shape[0]
    samples = np.tile(xeval, [Lambda_dim*2,1])

    translate = r*np.kron(np.eye(Lambda_dim), np.ones([num_xeval,1]))
    translate = np.append(translate, -translate, axis=0)
    
    samples += translate

    return samples

def pick_ffd_points(xeval, r):
    """

    Pick Lambda_dim points, for each xeval, for right side 
    finite diff grad approx.

    :param xeval: Points in :math:`\Lambda` to cluster points around
    :type xeval: :class:`np.ndarray` of shape (num_exval, Ldim)
    :param float r: Each side of the box will have length 2*r
    :rtype: :class:`np.ndarray` of shape (num_close*num_xeval, Ldim)
    :returns: Samples for centered finite difference stencil for 
        each point in xeval.


    """
    Lambda_dim = xeval.shape[1]
    num_xeval = xeval.shape[0]
    samples = np.tile(xeval, [Lambda_dim,1])

    translate = r*np.kron(np.eye(Lambda_dim), np.ones([num_xeval,1]))
    #translate = np.append(translate, -translate, axis=0)
    
    samples += translate

    return samples

def sample_diamond(centers, num_samples , r=None):
    """
    TO DO: come up with better variable names, vectorize
           split this into two methods, sample_simplex, simlex_to_diamond

    Uniformly sample a diamond (defined by 2^dim simplices).  Then scale
    each dimension according to rvec and translate the center to centers.
    Do this for each point in centers.

    :param centers: Points in :math:`\Lambda` to cluster samples around
    :type centers: :class:`np.ndarray` of shape (num_centers, Ldim)
    :param int num_samples: Number of samples in each diamond
    :param float r: The radius of the diamond, along each axis
    :rtype: :class:`np.ndarray` of shape (num_samples*num_centers, Ldim)
    :returns: Uniform random samples from a diamond around each center point


    """
    Lambda_dim = centers.shape[1]
    if r==None:
        rvec = np.ones([Lambda_dim,1])
    else:
        rvec = r*np.ones([Lambda_dim,1])

    x = np.zeros([1, centers.shape[1]])

    u = np.random.random([num_samples,1])
    b = u**(1./Lambda_dim)

    for j in range(centers.shape[0]):
        temp = np.random.random([num_samples,Lambda_dim-1])*np.tile(b, (1,Lambda_dim-1))
        temp = np.sort(temp,1)
        xtemp = np.zeros([num_samples, Lambda_dim])
        temp1 = np.zeros([num_samples, Lambda_dim+1])
        temp1[:,1:Lambda_dim] = temp
        temp1[:,Lambda_dim] = np.array(b).transpose()
        for i in range(1,Lambda_dim+1):
            xtemp[:,i-1] = temp1[:,i]-temp1[:,i-1]

        u_sign = 2*np.round(np.random.random([num_samples ,Lambda_dim])) - 1
        xtemp = xtemp*u_sign;

        for i in range(Lambda_dim):
            xtemp[:,i] = rvec[i]*xtemp[:,i]
        xtemp = xtemp + centers[j,:]
        x = np.append(x, xtemp, axis=0)

    return x[1:]


####################################

def test_function_poly(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    f = (x0-a0)**10 + ... + (xn-an)**10

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Test function evaluated at each point in x

    """
    Lambda_dim = x.shape[1]
    f = np.zeros([x.shape[0],1])
    np.random.seed(1)
    rand_constants = 2*np.random.random(Lambda_dim)

    for i in range(x.shape[1]):
        f[:,0] = f[:,0] + (x[:,i]-rand_constants[i])**10

    np.random.seed(seed=None)

    return f

def test_deriv_poly(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    fxi = 10*(xi-ai)**(10-1)

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Each partial derivative of test function evaluated
        at each point in x

    """
    Lambda_dim = x.shape[1]
    fxi = np.zeros(x.shape)
    np.random.seed(1)
    rand_constants = 2*np.random.random(Lambda_dim)

    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i==j:
                fxi[:,i] = 10*(x[:,j]-rand_constants[j])**(10-1)

    np.random.seed(seed=None)

    return fxi

def test_function_2d(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    f = (x0-a0)**10 + ... + (xn-an)**10

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Test function evaluated at each point in x

    """
    Lambda_dim = x.shape[1]
    f = np.zeros([x.shape[0],1])
    np.random.seed(1)
    #rand_constants = 2*np.random.random(Lambda_dim)


    f[:,0] = 5*((x[:,0]-1.0)**2 + (x[:,1]-1.0)**2)

    np.random.seed(seed=None)

    return f

def test_deriv_2d(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    fxi = 10*(xi-ai)**(10-1)

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Each partial derivative of test function evaluated
        at each point in x

    """
    Lambda_dim = x.shape[1]
    fxi = np.zeros(x.shape)
    np.random.seed(1)
    rand_constants = 2*np.random.random(Lambda_dim)

    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i==j:
                fxi[:,i] = 10*(x[:,j]-1)

    np.random.seed(seed=None)

    return fxi

def test_function_2d_sin(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    f = (x0-a0)**10 + ... + (xn-an)**10

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Test function evaluated at each point in x

    """
    Lambda_dim = x.shape[1]
    f = np.zeros([x.shape[0],1])
    #np.random.seed(1)
    #rand_constants = 2*np.random.random(Lambda_dim)


    f[:,0] = np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])

    #np.random.seed(seed=None)

    return f

def test_deriv_2d_sin(x):
    """

    Define a test_function to check the accuracy of the
    RBF and CDF gradient approximations.
    fxi = 10*(xi-ai)**(10-1)

    :param x: Points in :math:`\Lambda` to check grad approximation
    :type x: :class:`np.ndarray`
    :rtype: :class:`np.ndarray` of shape (x.shape[0], 1)
    :returns: Each partial derivative of test function evaluated
        at each point in x

    """
    Lambda_dim = x.shape[1]
    fxi = np.zeros(x.shape)
    #np.random.seed(1)
    rand_constants = 2*np.random.random(Lambda_dim)

    fxi[:,0] = np.pi*np.cos(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
    fxi[:,1] = np.pi*np.sin(np.pi*x[:,0])*np.cos(np.pi*x[:,1])

    #np.random.seed(seed=None)

    return fxi

##################################################
##################################################
##################################################
##################################################
# choose opt qois below

def chooseOptQoIsCOND(Grad_tensor, indexstart, indexstop):
    min_condnum = 1E10
    for i in range(indexstart,indexstop):
        for j in range(i+1,indexstop+1):
            sys.stdout.flush()
            sys.stdout.write("\ri = %i" % i)
            condnum_sum = 0
            for xe in range(Grad_tensor.shape[0]):
                condnum_sum += np.linalg.cond(Grad_tensor[xe,[i,j],:])
            current_condnum = condnum_sum/Grad_tensor.shape[0]

            if current_condnum < min_condnum:
                min_condnum = current_condnum
                qoi1index = i+1
                qoi2index = j+1
    
    return min_condnum, qoi1index, qoi2index

def chooseOptQoIsSVD(Grad_tensor, indexstart, indexstop):
    Lambda_dim = Grad_tensor.shape[2]
    num_xeval = Grad_tensor.shape[0]
    min_condnum = 1E10
    for i in range(indexstart,indexstop):
        for j in range(i+1,indexstop+1):
            sys.stdout.flush()
            sys.stdout.write("\ri = %i" % i)

            singvals = np.linalg.svd(Grad_tensor[:,[i,j],:], compute_uv=False)
            current_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/num_xeval

            if current_condnum < min_condnum:
                min_condnum = current_condnum
                qoi1index = i+1
                qoi2index = j+1
    
    return min_condnum, qoi1index, qoi2index




def chooseOptQoIs(Grad_tensor, indexstart, indexstop, num_qois_returned):
    Lambda_dim = Grad_tensor.shape[2]
    num_qois = indexstop-indexstart + 1
    num_xeval = Grad_tensor.shape[0]
    min_condnum = 1E10
    qoi_combinations = combinations(range(indexstart, indexstop+1), num_qois_returned)

    count = 0
    for qoi_set in qoi_combinations:
        count += 1
        sys.stdout.flush()
        sys.stdout.write("\rcount = %i" % count)

        # because I cant figure out how to vectorize np.linalg.cond
        singvals = np.linalg.svd(Grad_tensor[:,list(qoi_set),:], compute_uv=False)
        current_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/num_xeval

        if current_condnum < min_condnum:
            min_condnum = current_condnum
            qoiIndices = list(qoi_set)
    
    return min_condnum, qoiIndices

def chooseOptQoIsPAR(Grad_tensor, indexstart, indexstop, num_qois_returned):
    Lambda_dim = Grad_tensor.shape[2]
    num_qois = indexstop-indexstart + 1
    num_xeval = Grad_tensor.shape[0]
    min_condnum = 1E10
    #use range below instead of linspace
    qoi_combinations = combinations(range(indexstart, indexstop+1), num_qois_returned)

    count = 0
    for qoi_set in qoi_combinations:
        count += 1
        sys.stdout.flush()
        sys.stdout.write("\rcount = %i" % count)

        # because I cant figure out how to vectorize np.linalg.cond
        singvals = np.linalg.svd(Grad_tensor[:,list(qoi_set),:], compute_uv=False)
        current_condnum = np.sum(singvals[:,0]/singvals[:,-1], axis=0)/num_xeval

        if current_condnum < min_condnum:
            min_condnum = current_condnum
            qoiIndices = list(qoi_set)
    
    return min_condnum, qoiIndices





