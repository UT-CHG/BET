# Copyright (C) 2014-2019 The BET Development Team

"""
This module contains unittests for :mod:`~bet.Comm`
"""

import unittest
import bet.Comm as Comm
from pkgutil import iter_modules


class Test_comm_for_no_mpi4py(unittest.TestCase):
    """
    Test :class:`bet.Comm.comm_for_no_mpi4py`.
    """

    def setUp(self):
        self.comm = Comm.comm_for_no_mpi4py()

    def test_Get_size(self):
        self.assertEqual(self.comm.Get_size(), 1)

    def test_Get_rank(self):
        self.assertEqual(self.comm.Get_rank(), 0)

    def test_allgather(self):
        thing = list(range(4))
        self.assertEqual(self.comm.allgather(thing), [thing])

    def test_allreduce(self):
        thing = 4
        self.assertEqual(self.comm.allreduce(thing, op=None), thing)

    def test_bcast(self):
        thing = list(range(4))
        self.assertEqual(self.comm.bcast(thing, root=0), thing)

    def test_Allgather(self):
        thing = list(range(4))
        self.assertEqual(self.comm.Allgather(thing), thing)

    def test_Allreduce(self):
        thing1 = list(range(4))
        thing2 = list(range(4))
        self.assertEqual(self.comm.Allreduce(thing1, thing2,
                                             op=None), thing1)

    def test_Bcast(self):
        thing = list(range(4))
        self.assertEqual(self.comm.Bcast(thing, root=0), thing)

    def test_Scatter(self):
        thing1 = list(range(4))
        thing2 = list(range(4))
        self.assertEqual(self.comm.Scatter(thing1, thing2,
                                           root=0), thing1)


class Test_Comm(unittest.TestCase):
    """
    Test :mod:`bet.Comm`
    """

    def test(self):
        if 'mpi4py' in (name for loader, name, ispkg in iter_modules()):
            pass
        else:
            self.assertEqual(Comm.comm.size, 1)
            self.assertEqual(Comm.comm.rank, 0)


class Test_MPI_for_no_mpi4py(unittest.TestCase):
    """
    Test :class:`bet.Comm.MPI_fort_no_mpi4py`
    """

    def test(self):
        MPI_no = Comm.MPI_for_no_mpi4py()
        self.assertEqual(MPI_no.SUM, None)
        self.assertEqual(MPI_no.DOUBLE, float)
        self.assertEqual(MPI_no.INT, int)
