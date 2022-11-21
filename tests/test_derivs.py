# Test code for mathematical comps
import pytest
import numpy as np

from bad_package import *
from bad_package.fad import DualNumber

class TestDerivs():

    def test_scalarExp(self):
        # Compute exp(x^2) deriv evaluated at x = 2
        # Deriv should be 4 * exp(4)
        x1 = DualNumber(2)
        result = exp(x1**2)
        assert pytest.approx(4 * np.exp(4)) == result.dual

    def test_scalarSin(self):
        # Compute deriv of sin(2x) + 3 evaluated at x = pi
        # Deriv should be cos(2x) * 2 == 2 * cos(2pi) == 2
        x1 = DualNumber(pi)
        result = sin(2*x1) + 3
        assert pytest.approx(2 * np.cos(2*np.pi)) == result.dual

    def test_scalarLn(self):
        # Compute deriv of ln(2x^3) evaluated at x = 4
        # Deriv should be 3/x (simplified) == 3/4 == 0.75
        x1 = DualNumber(4)
        result = ln(2*x1**3)
        assert pytest.approx(3/4) == result.dual
