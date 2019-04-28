from dolfin import *
import numpy as np
import petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from meshDS import*
# initialize petsc
petsc4py.init()
petsc4py.PETSc.Sys.popErrorHandler()


class projectKL(object):

    """Docstring for projectKL. """

    def __init__(self, mesh):
        """TODO: to be defined1. """
        # create meshDS obect
        self._mesh = mesh
        self.domain = meshDS(mesh)
        self.c_volume_array, self.c_centroid_array = self.domain.getElemVCArray()
        self.node_to_elem = self.domain.getNodesToElem()
        self.flag = False

    def getCovMat(self, cov_expr):
        """TODO: Docstring for getCovMat.

        :cov_expr: Expression (dolfin) as a function of 
        :returns: covariance PETSC matrix cov_mat

        """
        # store the expression
        self.expr = cov_expr
        # create a PETSC matrix for cov_mat
        cov_mat = PETSc.Mat().create()
        cov_mat.setType('aij')
        cov_mat.setSizes(self.domain.getNodes(), self.domain.getNodes())
        cov_mat.setUp()

        # scalar valued function is evaluated in this variable
        cov_ij = np.empty((1), dtype=float)
        # the points to evalute the expression
        xycor = np.empty((4), dtype=float)

        print('---------------------------')
        print('---------------------------')
        print(' Building Covariance Matrix')
        print('---------------------------')
        print('---------------------------')
        # Loop through global nodes and build the matrix for i < j because of symmetric nature.
        for node_i in range(0, self.domain.getNodes()):
            # global node node_i
            for node_j in range(node_i, self.domain.getNodes()):
                # global node node_j
                temp_cov_ij = 0
                for elem_i in self.node_to_elem[node_i]:
                    # elem_i : element attached to node_i
                    # x1 : x co-ordinate of the centroid of element elem_i
                    x1 = self.c_centroid_array[elem_i].x()
                    # y1 : x co-ordinate of the centroid of element elem_i
                    y1 = self.c_centroid_array[elem_i].y()
                    for elem_j in self.node_to_elem[node_j]:
                        # elem_j : element attached to node_j
                        # x2 : x co-ordinate for the centroid of element elem_j
                        x2 = self.c_centroid_array[elem_j].x()
                        # y2 : y co-ordinate for the centroid of element elem_j
                        y2 = self.c_centroid_array[elem_j].y()
                        xycor[0] = x1
                        xycor[1] = x2
                        xycor[2] = y1
                        xycor[3] = y2
                        # evaluate the expression
                        cov_expr.eval(cov_ij, xycor)
                        if cov_ij[0] > 0:
                            temp_cov_ij += (1.0/3)*(1.0/3)*cov_ij[0]*self.c_volume_array[elem_i] * \
                                self.c_volume_array[elem_j]
                            cov_mat.setValue(node_i, node_j, temp_cov_ij)
                            cov_mat.setValue(node_j, node_i, temp_cov_ij)
        cov_mat.assemblyBegin()
        cov_mat.assemblyEnd()
        print('---------------------------')
        print('---------------------------')
        print(' Finished Covariance Matrix')
        print('---------------------------')
        print('---------------------------')

        return cov_mat

    def _getBMat(self):
        """TODO: Docstring for getBmat. We are solving for CX = BX where C is the covariance matrix
        and B is just a mass matrix. Here we assemble B. This is a private function. DONT call this 
        unless debuging.

        :returns: PETScMatrix B 
        """

        # B matrix is just a mass matrix, can be easily assembled through fenics
        # however, the ordering in fenics is not the mesh ordering. so we build a temp matrix
        # then use the vertex to dof map to get the right ordering interms of our mesh nodes
        V = FunctionSpace(self._mesh, "CG", 1)
        # Define basis and bilinear form
        u = TrialFunction(V)
        v = TestFunction(V)
        a = u*v*dx
        B_temp = assemble(a)

        B = PETSc.Mat().create()
        B.setType('aij')
        B.setSizes(self.domain.getNodes(), self.domain.getNodes())
        B.setUp()

        B_ij = B_temp.array()

        v_to_d_map = vertex_to_dof_map(V)

        print('---------------------------')
        print('---------------------------')
        print(' Building Mass Matrix ')
        print('---------------------------')
        print('---------------------------')
        for node_i in range(0, self.domain.getNodes()):
            for node_j in range(node_i, self.domain.getNodes()):
                B_ij_nodes = B_ij[v_to_d_map[node_i], v_to_d_map[node_j]]
                if B_ij_nodes > 0:
                    B.setValue(node_i, node_j, B_ij_nodes)
                    B.setValue(node_j, node_i, B_ij_nodes)

        B.assemblyBegin()
        B.assemblyEnd()
        print('---------------------------')
        print('---------------------------')
        print(' Finished Mass Matrix ')
        print('---------------------------')
        print('---------------------------')
        return B

    def projectCovToMesh(self, num_kl, cov_expr):
        """TODO: Docstring for projectCovToMesh. Solves CX = BX where C is the covariance matrix
        :num_kl : number of kl exapansion terms needed 
        :returns: TODO

        """
        # turn the flag to true
        self.flag = True
        # get C,B matrices
        C = self.getCovMat(cov_expr)
        B = self._getBMat()
        # Solve the generalized eigenvalue problem
        eigensolver = SLEPc.EPS()
        eigensolver.create()
        eigensolver.setOperators(C, B)
        eigensolver.setDimensions(num_kl)
        eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eigensolver.setFromOptions()
        eigensolver.solve()
        # Get the number of eigen values that converged.
        #nconv = eigensolver.get_number_converged()

        # Get N eigenpairs where N is the number of KL expansion and check if N < nconv otherwise you had
        # really bad matrix

        # create numpy array of vectors and eigenvalues
        self.eigen_funcs = np.empty((num_kl), dtype=object)
        self.eigen_vals = np.empty((num_kl), dtype=float)

        # store the eigenvalues and eigen functions
        V = FunctionSpace(self._mesh, "CG", 1)
        x_real = PETSc.Vec().create()
        x_real.setSizes(self.domain.getNodes())
        x_real.setUp()
        x_real.setValues(np.arange(0, self.domain.getNodes()),
                         np.zeros(self.domain.getNodes()))
#        for i in range(0, self.domain.getNodes()):
#            x_real.setValue(i, 0, 0.)

        for eigen_pairs in range(0, num_kl):
            lam = eigensolver.getEigenpair(eigen_pairs, x_real)
            self.eigen_funcs[eigen_pairs] = Function(V)
            # use dof_to_vertex map to map values to the function space
            self.eigen_funcs[eigen_pairs].vector()[:] = x_real.getValues(
                dof_to_vertex_map(V).astype('int32'))
            # divide by norm to make the unit norm again
            self.eigen_funcs[eigen_pairs].vector()[:] = self.eigen_funcs[eigen_pairs].vector()[:] / \
                norm(self.eigen_funcs[eigen_pairs])
            self.eigen_vals[eigen_pairs] = lam.real
