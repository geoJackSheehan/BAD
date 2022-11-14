# Test code for src/bad_package/elementary_functions.py

from typing import Type
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad.fad import DualNumber

class TestElementaryFunctions():

    def test_exp():
        # still in progress
        assert isinstance(exp(DualNumber(1, 1)), DualNumber)
        with pytest.raises(TypeError):
            exp('3')
            exp(['3'])

        x = DualNumber(2, 2)
        y = exp(x)
        num = 5
        result = exp(5)

        assert y.real == np.exp(2)
        assert result == np.exp(5)

    def test_ln():
        pass

    def test_logbase():
        pass

    def test_sin():
        pass

    def test_cos():
        pass

    def test_tan():
        pass

    def test_csc():
        pass

    def test_sec():
        pass

    def test_cot():
        pass

    def test_sinh():
        pass

    def test_cosh():
        pass

    def test_tanh():
        pass

    def test_arcsin():
        pass

    def test_arccos():
        pass

    def test_arctan():
        pass

    def test_arcsinh():
        pass

    def test_arccosh():
        pass

    def test_arctanh():
        pass

    def test_sqrt():
        pass
