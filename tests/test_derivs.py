# Test code for mathematical comps
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad.fad import DualNumber

class TestDerivs():

    def test_scalarDeriv(self):
        # Compute exp(x^2) deriv evaluated at x = 2
        # Deriv should be 4 * exp(4)
        x1 = DualNumber(2)
        result = exp(x1**2)
        assert pytest.approx(2 * np.e) == result.dual

        # Compute deriv of sin(2x) + 3 evaluated at x = pi
        # Deriv should be cos(2x) * 2 == 2 * cos(2pi) == 2
        x2 = DualNumber(pi)
        result2 = sin(2*x2) + 3
        assert pytest.approx(2 * np.cos(2*x2)) == result.dual

        # Compute deriv of ln(2x^3) evaluated at x = 4
        # Deriv should be 3/x (simplified) == 3/4 == 0.75
        x3 = DualNumber(4)
        result3 = ln(2*x3**3)
        assert pytest.approx(3 / 4) == result.dual