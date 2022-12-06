# Test code for ad interface
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad import DualNumber
from bad_package.interface import AutoDiff
from bad_package.interface import ReverseAD

class TestADInterface():

    def test_scalar_get_primal(self):
        def func(x):
            return 4*x + 3
        x = np.array([2])
        ad = AutoDiff(func, x)
        ad.compute()
        result = ad.get_primal()
        assert pytest.approx(11) == result

    def test_scalar_get_jacobian(self):
        def func(x):
            return 4*x + 3
        x = np.array([2])
        ad = AutoDiff(func, x)
        ad.compute()
        result = ad.get_jacobian()
        assert pytest.approx([4]) == result

    def test_vector_get_primal(self):
        def func(x):
            return x[0]**2 + 3*x[1] + 5
        x = np.array([1, 2])
        ad = AutoDiff(func, x)
        ad.compute()
        result = ad.get_primal()
        assert pytest.approx(12) == result
        
    def test_vector_get_jacobian(self):
        def func(x):
            return x[0]**2 + 3*x[1] + 5
        x = np.array([1, 2])
        ad = AutoDiff(func, x)
        ad.compute()
        result = ad.get_jacobian()
        assert pytest.approx([2,3]) == result

    def test_scalar_get_jacobian_RM(self):
        def func(x):
            return 4*x
        x = np.array([2])
        rm = ReverseAD(func, x)
        rm.compute()
        result = rm.get_jacobian()
        assert pytest.approx([4]) == result