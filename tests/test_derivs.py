# Test code for mathematical comps
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad.fad import DualNumber

class TestDerivs():

    def test_scalarDeriv(self):
        x1 = DualNumber(1)
        result = exp(x1**2)

        assert pytest.approx(2 * e) == result.dual


        