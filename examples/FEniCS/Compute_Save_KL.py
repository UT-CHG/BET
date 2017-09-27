# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
import numpy as np
from dolfin import *
from meshDS import meshDS
from projectKL import projectKL
from poissonRandField import solvePoissonRandomField
import scipy.io as sio
import argparse

# This has been changed from a function to a script which can be run in
# simple docker container. The function is defined below, but as a script,
# this file takes in-line arguments for numKL. Default value is 2.

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('Kval', type=int, nargs='?', default='2',
                    help='an integer for the number of KL terms')

nK = parser.parse_args()
print(nK.Kval)
print("which is type: " +str(type(nK.Kval)))

def computeSaveKL(numKL):
    '''
    ++++++++++++++++ Steps in Computing the Numerical KL Expansion ++++++++++
    We proceed by loading the mesh and defining the function space for which
    the eigenfunctions are defined upon.

    Then, we define the covariance kernel which requires correlation lengths
    and a standard deviation.

    We then compute and save the terms in a truncated KL expansion.
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    '''

    # Step 1: Set up the Mesh and Function Space

    mesh = Mesh("Lshaped.xml")

    # initialize the mesh to generate connectivity
    mesh.init()

    # Random field is projected on the space of Hat functions in the mesh
    V = FunctionSpace(mesh, "CG", 1)

    # Step 2: Project covariance in the mesh and get the eigenfunctions

    # Initialize the projectKL object with the mesh
    Lmesh = projectKL(mesh)

    # Create the covariance expression to project on the mesh.
    etaX = 10.0
    etaY = 10.0
    C = 1

    # Pick your favorite covariance. Popular choices are Gaussian (of course),
    # Exponential, triangular (has finite support which is nice). Check out
    # Ghanem and Spanos' book for more classical options.

    # A Gaussian Covariance
    '''
    cov = Expression("C*exp(-((x[0]-x[1]))*((x[0]-x[1]))/ex - \
      ((x[2]-x[3]))*((x[2]-x[3]))/ey)",
             ex=etaX,ey=etaY, C=C)
    '''
    # An Exponential Covariance
    cov = Expression("C*exp(-fabs(x[0]-x[1])/ex - fabs(x[2]-x[3])/ey)",ex=etaX,ey=etaY, C=C, \
	degree = 10)

    # Solve the discrete covariance relation on the mesh
    Lmesh.projectCovToMesh(numKL,cov)

    # Get the eigenfunctions and eigenvalues
    eigen_val = Lmesh.eigen_vals

    eigen_func_mat = np.zeros((numKL, Lmesh.eigen_funcs[0].vector().array().size))
    for i in range(0,numKL):
        eigen_func_mat[i,:] = Lmesh.eigen_funcs[i].vector().array()

    kl_mdat = dict()
    kl_mdat['KL_eigen_funcs'] = eigen_func_mat
    kl_mdat['KL_eigen_vals'] = eigen_val

    sio.savemat("KL_expansion", kl_mdat)


# added code which runs KL function defined above with nK terms
computeSaveKL(nK.Kval)
print("Were the KL_expansions saved?")
