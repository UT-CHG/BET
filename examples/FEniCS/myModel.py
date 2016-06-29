# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
import numpy as np
from dolfin import *
from meshDS import meshDS
from projectKL import projectKL
from poissonRandField import solvePoissonRandomField
import scipy.io as sio

def my_model(parameter_samples):

    # We proceed by loading the mesh and defining the function space for which
    # the eigenfunctions from Compute_Save_KL were computed.

    # Step 1: Set up the Mesh and Function Space
    mesh = Mesh("Lshaped.xml")

    # initialize the mesh to generate connectivity
    mesh.init()

    # Random field is projected on the space of Hat functions in the mesh
    V = FunctionSpace(mesh, "CG", 1)

    # Load the KL expansion information
    KL_mdat = sio.loadmat("KL_expansion")
    KL_eigen_funcs = KL_mdat['KL_eigen_funcs']
    KL_eigen_vals = KL_mdat['KL_eigen_vals']

    # number of parameter samples
    numSamples = parameter_samples.shape[0]

    # number of KL expansion terms.
    numKL = parameter_samples.shape[1]

    # the samples are the coefficients of the KL expansion typically denoted by xi_k
    xi_k = parameter_samples

    '''
    ++++++++++++++++ Steps in Solving Poisson with the KL fields ++++++++++++
    First set up the necessary variables and boundary conditions for the
    problem.

    Then create the QoI maps defined by average values over some part of the
    physical domain.

    Loop through the sample fields and create the permeability defined by the
    exponential of the KL field (i.e., the KL expansion represents the log of
    the permeability).

    For each sample field, call the PoissonRandomField solver.
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''
    # permeability
    perm_k = Function(V)

    # Create Boundary Conditions -- Dirichlet on left and bottom boundary.
    # Left Dirichlet Bc
    def left_boundary(x,on_boundary):
        """TODO: Docstring for left_boundary.

        :x: TODO
        :on_boundary: TODO
        :returns: TODO

        """
        tol = 1e-14
        return on_boundary and abs(x[0]) < tol
    Gamma_0 = DirichletBC(V,Constant(0.0),left_boundary)

    def bottom_boundary(x,on_boundary):
        """TODO: Docstring for left_boundary.

        :x: TODO
        :on_boundary: TODO
        :returns: TODO

        """
        tol = 1e-14
        return on_boundary and abs(x[1]) < tol
    Gamma_1 = DirichletBC(V,Constant(0.0),bottom_boundary)
    bcs = [Gamma_0,Gamma_1]

    # Setup the QoI class
    class CharFunc(Expression):
      def __init__(self, region):
        self.a = region[0]
        self.b = region[1]
        self.c = region[2]
        self.d = region[3]
      def eval(self, v, x):
        v[0] = 0
        if (x[0] >= self.a) & (x[0] <= self.b) & (x[1] >= self.c) & (x[1] <= self.d):
          v[0] = 1
        return v

    # Define the QoI maps
    Chi_1 = CharFunc([0.75, 1.25, 7.75, 8.25])
    Chi_2 = CharFunc([7.75, 8.25, 0.75, 1.25])

    QoI_samples = np.zeros([numSamples,2])

    # For each sample solve the PDE
    f = Constant(-1.0) # forcing of Poisson

    for i in range(0, numSamples):

        print "Sample point number: %g" % i

        # create a temp array to store logPerm as sum of KL expansions
        # logPerm is log permeability
        logPerm = np.zeros((mesh.num_vertices()), dtype=float)
        for kl in range(0, numKL):
            logPerm += xi_k[i, kl] * \
                       np.sqrt(KL_eigen_vals[0,kl]) * KL_eigen_funcs[kl,:]

        # permiability is the exponential of log permeability logPerm
        perm_k_array = 0.1 + np.exp(logPerm)

        # map permiability values to the function space object perm_k
        perm_k.vector()[:] = perm_k_array

        # solve Poisson with this random field using FEM
        u = solvePoissonRandomField(perm_k, mesh, 1, f, bcs)

        # Compute QoI
        QoI_samples[i, 0] = assemble(u * Chi_1 * dx)
        QoI_samples[i, 1] = assemble(u * Chi_2 * dx)


    return QoI_samples