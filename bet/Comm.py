"""
This module provides a workaround for people without mpi4py installed 
to run BET.
"""

class comm_for_no_mpi4py:
    def __init__(self):
        pass
    def Get_size(self):
        return 1
    def Get_rank(self):
        return 0 
    def allgather(self,val):
        return val
    def allreduce(self,val1, op=None):
        return val1
    def bcast(self,val, root=0):
        return val   
    def Allgather(self,val):
        return val
    def Allreduce(self,val1, val2, op=None):
        return val1
    def Bcast(self,val, root=0):
        return val
    def Scatter(self, val1, val2, root=0):
        return val1

class MPI_for_no_mpi4py:
    def __init__(self):
        self.SUM = None
        self.DOUBLE = float
        self.INT = int
        
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    MPI = MPI_for_no_mpi4py()
    comm = comm_for_no_mpi4py()
    
size = comm.Get_size()
rank = comm.Get_rank()
