# Copyright (C) 2014-2015 Lindley Graham and Steven Mattis

"""
This module provides a workaround for people without mpi4py installed
to run BET.
"""


class comm_for_no_mpi4py(object):

    """
    Provides a fake MPI.COMM_WORLD implementation so that the user need not
    install mpi4py.
    """

    def __init__(self):
        """
        Initialization
        """
        pass

    def Get_size(self):
        """
        :rtype: int
        :returns: 1
        """
        return 1

    def Get_rank(self):
        """
        :rtype: int
        :returns 0
        """
        return 0

    def allgather(self, val):
        """
        :param object val: object to allgather
        :rtype: object
        :returns: val
        """
        return val

    def allreduce(self, val1, op=None):
        """
        :param object val1: object to allreduce
        :rtype: object
        :returns: val1
        """
        return val1

    def bcast(self, val, root=0):
        """
        :param object val: object to broadcast
        :param int root: 0
        :rtype: object
        :returns: val
        """
        return val

    def Allgather(self, val):
        """
        :param object val: object to Allgather
        :rtype: object
        :returns: val
        """
        return val

    def Allreduce(self, val1, val2, op=None):
        """
        :param object val1: object to Allreduce
        :param object val2: object to Allreduce
        :param op: None
        :rtype: object
        :returns: val1
        """
        return val1

    def Bcast(self, val, root=0):
        """
        :param object val: object to gather
        :param int root: 0
        :rtype: object
        :returns: val
        """
        return val

    def Scatter(self, val1, val2, root=0):
        """
        :param object val1: object to Scatter
        :param object val2: object to Scatter
        :param int root: 0
        :rtype: object
        :returns: val1
        """
        return val1


class MPI_for_no_mpi4py(object):

    """
    Provides a fake MPI implementation so that the user need not install
    mpi4py.
    """

    def __init__(self):
        """
        Initialization
        """
        self.SUM = None
        self.DOUBLE = float
        self.INT = int
        self.BOOL = bool

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    MPI = MPI_for_no_mpi4py()
    comm = comm_for_no_mpi4py()

size = comm.Get_size()
rank = comm.Get_rank()
