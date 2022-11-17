# Test code for src/bad_package/elementary_functions.py

from typing import Type
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad.fad import DualNumber

class TestElementaryFunctions():

    def test_exp(self):
        assert isinstance(exp(DualNumber(1, 1)), DualNumber)
        with pytest.raises(TypeError):
            exp('3')
            exp(['3'])

        x = DualNumber(2, 2)
        y = exp(x)
        num = 5
        result = exp(5)
        self.assertAlmostEqual(np.exp(2), y.real)
        self.assertAlmostEqual(np.exp(5), result)

    def test_ln(self):
        assert isinstance(ln(DualNumber(2, 2)), DualNumber)
        with pytest.raises(TypeError):
            ln('text')

        with pytest.raises(ArithmeticError):
            ln(0)
            ln(-1)

        # x = DualNumber(2, 2)
        # y = ln(x)
        #
        # self.assertAlmostEqual(np.ln(2), y.real)

    def test_logbase(self):
        assert isinstance(log(DualNumber(2, 5)), DualNumber)
        x = DualNumber(2, 5)
        result = log(x)
        self.assertAlmostEqual(np.log(2)/np.log(5), result.real)
        self.assertAlmostEqual(1/(2*np.log(5))*5, result.dual)

    def test_sin(self):
        assert isinstance(sin(DualNumber(2, 2)), DualNumber)
        x = DualNumber(5, 2)
        result = sin(x)
        self.assertAlmostEqual(np.sin(5), result.real)
        self.assertAlmostEqual(2*np.cos(5), result.dual)

        x_neg = DualNumber(-5, 2)
        result_neg = sin(x_neg)
        self.assertAlmostEqual(np.sin(-5), result_neg.real)
        self.assertAlmostEqual(2*np.cos(-5), result_neg.dual)

    def test_cos(self):
        assert isinstance(cos(DualNumber(2, 2)), DualNumber)
        x = DualNumber(8, 3)
        result = cos(x)
        self.assertAlmostEqual(np.cos(8), result.real)
        self.assertAlmostEqual(3*(-np.sin(8)), result.dual)

        x_neg = DualNumber(-2, 2)
        result_neg = cos(x_neg)
        self.assertAlmostEqual(np.cos(-2), result_neg.real)
        self.assertAlmostEqual(2*(-np.sin(-2)), result_neg.dual)

    def test_tan(self):
        assert isinstance(tan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 5)
        result = tan(x)
        self.assertAlmostEqual(np.tan(2), result.real)
        self.assertAlmostEqual(5/(np.cos(2)**2), result.dual)

        x_neg = DualNumber(-0.5, -3)
        result_neg = tan(x_neg)
        self.assertAlmostEqual(np.tan(-0.5), result_neg.real)
        self.assertAlmostEqual(-3/(np.cos(-0.3)**2), result_neg.dual)

    def test_csc(self):
        # csc'(x) = -csc(x)cot(x)
        assert isinstance(csc(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = csc(x)
        self.assertAlmostEqual(1/sin(2), result.real)
        self.assertAlmostEqual(-3*np.csc(2)*cot(2), result.dual)

    def test_sec(self):
        # sec'(x) = sec(x)tan(x)
        assert isinstance(sec(DualNumber(2, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = sec(x)
        self.assertAlmostEqual(1/cos(2), result.real)
        self.assertAlmostEqual(3*np.sec(2)*np.tan(2), result.dual)

    def test_cot(self):
        # cot'(x) = -csc^2(x)
        assert isinstance(cot(DualNumber(2, 2)), DualNumber)
        x = DualNumber(4, 3)
        result = cot(x)
        self.assertAlmostEqual(1/tan(4), result.real)
        self.assertAlmostEqual(-3*(np.csc(4)**2), result.dual)

    def test_sinh(self):
        assert isinstance(sinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(-0.25, 1.5)
        result = sinh(x)
        self.assertAlmostEqual(np.sinh(-0.25), result.real)
        self.assertAlmostEqual(1.5*np.cosh(-0.25), result.dual)

    def test_cosh(self):
        assert isinstance(cosh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 5)
        result = cosh(x)
        self.assertAlmostEqual(np.cosh(2), result.real)
        self.assertAlmostEqual(5*np.sinh(2), result.dual)

    def test_tanh(self):
        # tanh'(x) = 1 - tanh^2(x)
        assert isinstance(tanh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(1.1, 2.2)
        result = tanh(x)
        self.assertAlmostEqual(np.tanh(1.1), result.real)
        self.assertAlmostEqual(2.2 * (1 - (np.tanh(1.1)**2)), result.dual)

    def test_arcsin(self):
        assert isinstance(arcsin(DualNumber(1, 1)), DualNumber)
        x = DualNumber(0.25, -5)
        result = arcsin(x)
        self.assertAlmostEqual(np.arcsin(0.25), result.real)
        self.assertAlmostEqual(-5/np.sqrt(1 - 0.25**2), result.dual)

    def test_arccos(self):
        assert isinstance(arccos(DualNumber(2, 3)), DualNumber)
        x = DualNumber(0.75, -2)
        result = arccos(x)
        self.assertAlmostEqual(np.arccos(0.75), result.real)
        self.assertAlmostEqual((-1)*(-2)/(np.sqrt((1-(0.75)**2))), result.dual)

    def test_arctan(self):
        assert isinstance(arctan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = arctan(x)
        self.assertAlmostEqual(np.arctan(2), result.real)
        self.assertAlmostEqual(3*(1/(1+(2**2))))

    def test_arcsinh(self):
        assert isinstance(arcsinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = arcsinh(x)
        self.assertAlmostEqual(np.arcsinh(2), result.real)
        self.assertAlmostEqual(3/(np.sqrt(2**2)+1), result.dual)

    def test_arccosh(self):
        assert isinstance(arccosh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(0.2, 0.3)
        result = arccosh(x)
        self.assertAlmostEqual(np.arccosh(0.2), result.real)
        self.assertAlmostEqual(0.2/(np.sqrt(0.2**2) - 1), result.dual)

    def test_arctanh(self):
        assert isinstance(arctanh(DualNumber(1, 3)), DualNumber)
        x = DualNumber(3, 5)
        result = arctanh(x)
        self.assertAlmostEqual(np.arctanh(3), result.real)
        self.assertAlmostEqual(5/(1 - 3**2), result.dual)

    def test_sqrt(self):
        assert isinstance(sqrt(DualNumber(4, 2)), DualNumber)
        x = DualNumber(4, -1)
        result = sqrt(x)
        self.assertAlmostEqual(np.sqrt(4), result.real)
        self.assertAlmostEqual((-1 * 0.5)*np.power(4, -0.5),result.dual)
