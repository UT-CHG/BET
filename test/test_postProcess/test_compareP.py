# Copyright (C) 2014-2019 The BET Development Team

import numpy as np
import numpy.testing as nptest
import unittest, os, glob
import bet.sample as sample
import bet.postProcess.compareP as compP
#import bet.util as util
#from bet.Comm import comm, MPI

#local_path = os.path.join(os.path.dirname(bet.__file__), "/test")
local_path = ''

class Test_metrization_simple(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        self.num1, self.num2, self.num3 = 100, 100, 500
        values1 = np.ones((self.num1, self.dim))
        values2 = np.ones((self.num2, self.dim))
        values3 = np.ones((self.num3, self.dim))
        self.integration_set = sample.sample_set(dim=self.dim)
        self.left_set = sample.sample_set(dim=self.dim)
        self.right_set = sample.sample_set(dim=self.dim)
        self.integration_set.set_values(values1)
        self.left_set.set_values(values2)
        self.right_set.set_values(values3)
        self.mtrc = compP.metrization(sample_set_left=self.left_set,
                                      sample_set_right=self.right_set,
                                      integration_sample_set=self.integration_set)
        
     
    def test_dimension(self):
        """
        Check that improperly setting dimension raises warning.
        """
        dim = self.dim+1
        values = np.ones((200, dim))
        integration_set = sample.sample_set(dim=dim)
        integration_set.set_values(values)
        try: 
            self.mtrc = compP.metrization(sample_set_left=self.left_set,
                                      sample_set_right=self.right_set,
                                      integration_sample_set=self.integration_set)
        except ValueError as e: # setting wrong shapes should raise this error
            pass
 
    def test_no_sample_set(self):
        test_set  = sample.sample_set(dim=self.dim)
        try:
            self.mtrc = compP.metrization(test_set, None)
        except:
            pass
        try:
            self.mtrc = compP.metrization(None, test_set)
        except:
            pass

    def test_set_ptr_left(self):
        """
        Test setting left io ptr
        """
        #TODO be careful if we change Kdtree
        self.mtrc.set_io_ptr_left(globalize=True)
        self.mtrc.get_io_ptr_left()
        self.mtrc.set_io_ptr_left(globalize=False)
        self.mtrc.get_io_ptr_left()
        self.mtrc.globalize_ptrs()
    
    def test_set_ptr_right(self):
        """
        Test setting right io ptr
        """
        #TODO be careful if we change Kdtree
        self.mtrc.set_io_ptr_right(globalize=True)
        self.mtrc.get_io_ptr_right()
        self.mtrc.set_io_ptr_right(globalize=False)
        self.mtrc.get_io_ptr_right()
        self.mtrc.globalize_ptrs()


