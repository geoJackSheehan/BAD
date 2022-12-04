# Test code for src/bad_package/elementary_functions.py

from typing import Type
import pytest
import numpy as np

from bad_package.elementary_functions import *
from bad_package.fad import DualNumber
from bad_package.rad import ReverseMode

class TestElementaryFunctions():

    def test_exp(self):
        # DualNumber
        assert isinstance(exp(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 3)
        y = exp(x)
        assert np.exp(2) == y.real
        assert 3*np.exp(2) == y.dual

        # ReverseMode
        assert isinstance(exp(ReverseMode(1)), ReverseMode)
        x = ReverseMode(2)
        y = exp(x)
        assert np.exp(2) == y.real

        # General
        with pytest.raises(TypeError):
            exp('3')
            exp(['3'])
            exp([])
        

    def test_ln(self):
        # DualNumber
        assert isinstance(ln(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        y = ln(x)
        assert ln(2) == y.real
        assert 3/2 == y.dual

        # ReverseMode
        assert isinstance(ln(ReverseMode(2)), ReverseMode)
        x = ReverseMode(2)
        y = ln(x)
        assert ln(2) == y.real

        # General
        with pytest.raises(TypeError):
            ln('text')

        with pytest.raises(ArithmeticError):
            ln(0)
            ln(-1)

        with pytest.raises(ArithmeticError):
            ln(DualNumber(-1, -1))
            ln(ReverseMode(-1))


    def test_logbase(self):
        # DualNumber
        assert isinstance(logBase(DualNumber(2, 5), np.e), DualNumber)
        x = DualNumber(2, 5)
        result = logBase(x, np.e)
        assert pytest.approx(np.log(2)/np.log(np.e)) == result.real
        assert pytest.approx(5*(1/(2*np.log(np.e)))) == result.dual

        # ReverseMode
        assert isinstance(logBase(ReverseMode(2), np.e), ReverseMode)
        x = ReverseMode(2)
        result = logBase(x, np.e)
        assert pytest.approx(np.log(2)/np.log(np.e)) == result.real

        # General
        with pytest.raises(ArithmeticError):
            logBase(DualNumber(0, 0), 2)
            logBase(0, 1)

        with pytest.raises(TypeError):
            logBase(64, '2')


    def test_sin(self):
        # DualNumber
        assert isinstance(sin(DualNumber(2, 2)), DualNumber)
        x = DualNumber(5, 2)
        result = sin(x)
        assert pytest.approx(np.sin(5)) == result.real
        assert pytest.approx(2*np.cos(5)) == result.dual

        x_neg = DualNumber(-5, 2)
        result_neg = sin(x_neg)
        assert pytest.approx(np.sin(-5)) == result_neg.real
        assert pytest.approx(2*np.cos(-5)) == result_neg.dual

        # ReverseMode
        assert isinstance(sin(ReverseMode(2)), ReverseMode)
        x = ReverseMode(5)
        result = sin(x)
        assert pytest.approx(np.sin(5)) == result.real

        x_neg = ReverseMode(-5)
        result_neg = sin(x_neg)
        assert pytest.approx(np.sin(-5)) == result_neg.real


    def test_cos(self):
        # DualNumber 
        assert isinstance(cos(DualNumber(2, 2)), DualNumber)
        x = DualNumber(8, 3)
        result = cos(x)
        assert np.cos(8) == result.real
        assert pytest.approx(3*(-np.sin(8))) == result.dual

        x_neg = DualNumber(-2, 2)
        result_neg = cos(x_neg)
        assert np.cos(-2) == result_neg.real
        assert pytest.approx(2*(-np.sin(-2))) == result_neg.dual

        # ReverseMode
        assert isinstance(cos(ReverseMode(2)), ReverseMode)
        x = ReverseMode(8)
        result = cos(x)
        assert np.cos(8) == result.real

        x_neg = ReverseMode(-2)
        result_neg = cos(x_neg)
        assert np.cos(-2) == result_neg.real


    def test_tan(self):
        assert isinstance(tan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 5)
        result = tan(x)
        assert np.tan(2) == result.real
        assert pytest.approx(5/(np.cos(2)**2)) == result.dual

        x_neg = DualNumber(-0.5, -3)
        result_neg = tan(x_neg)
        assert np.tan(-0.5) == result_neg.real
        assert pytest.approx(-3/(np.cos(-0.5)**2)) == result_neg.dual

        with pytest.raises(ArithmeticError):
            tan(DualNumber(pi/2))


    def test_csc(self):
        # csc'(x) = -csc(x)cot(x)
        assert isinstance(csc(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = csc(x)
        assert 1/sin(2) == result.real
        assert pytest.approx(-3*(1/np.sin(2))*(1/np.tan(2))) == result.dual

        with pytest.raises(ArithmeticError):
            csc(DualNumber(pi))


    def test_sec(self):
        # sec'(x) = sec(x)tan(x)
        assert isinstance(sec(DualNumber(2, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = sec(x)
        assert 1/cos(2) == result.real
        assert pytest.approx(3*(1/np.cos(2))*np.tan(2)) == result.dual

        with pytest.raises(ArithmeticError):
            sec(DualNumber(pi/2))


    def test_cot(self):
        # cot'(x) = -csc^2(x)
        assert isinstance(cot(DualNumber(2, 2)), DualNumber)
        x = DualNumber(4, 3)
        result = cot(x)
        assert 1/tan(4) == result.real
        assert pytest.approx(-3*((1/np.sin(4))**2)) == result.dual

        with pytest.raises(ArithmeticError):
            cot(DualNumber(pi))


    def test_sinh(self):
        assert isinstance(sinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(-0.25, 1.5)
        result = sinh(x)
        assert np.sinh(-0.25) == result.real
        assert pytest.approx(1.5*np.cosh(-0.25)) == result.dual


    def test_cosh(self):
        assert isinstance(cosh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 5)
        result = cosh(x)
        assert np.cosh(2) == result.real
        assert pytest.approx(5*np.sinh(2)) == result.dual


    def test_tanh(self):
        # tanh'(x) = 1 - tanh^2(x)
        assert isinstance(tanh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(.1, .2)
        result = tanh(x)
        assert pytest.approx(np.tanh(.1)) == result.real
        assert pytest.approx(.2 * (1/(np.cosh(.1)))**2) == result.dual


    def test_arcsin(self):
        assert isinstance(arcsin(DualNumber(0.9, 1)), DualNumber)
        x = DualNumber(0.25, 5)
        result = arcsin(x)
        assert np.arcsin(0.25) == result.real
        assert pytest.approx(5/np.sqrt(1 - 0.25**2)) ==  result.dual

        with pytest.raises(ArithmeticError):
            arcsin(DualNumber(-1, -1))
            arcsin(0)


    def test_arccos(self):
        assert isinstance(arccos(DualNumber(0.9, 3)), DualNumber)
        x = DualNumber(0.75, -.2)
        result = arccos(x)
        assert np.arccos(0.75) == result.real
        assert pytest.approx((-1)*(-.2)/(np.sqrt((1-(0.75)**2)))) == result.dual

        with pytest.raises(ArithmeticError):
            arccos(DualNumber(-1, -1))
            arccos(0)


    def test_arctan(self):
        assert isinstance(arctan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = arctan(x)
        assert np.arctan(2) == result.real
        assert pytest.approx(3*(1/(1+(2**2)))) == result.dual


    def test_arcsinh(self):
        assert isinstance(arcsinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = arcsinh(x)
        assert np.arcsinh(2) == result.real
        assert pytest.approx(3/(np.sqrt(2**2 + 1))) == result.dual


    def test_arccosh(self):
        assert isinstance(arccosh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 0.3)
        result = arccosh(x)
        assert np.arccosh(2) == result.real
        assert pytest.approx(0.3/(np.sqrt(2**2 - 1))) == result.dual

        with pytest.raises(ArithmeticError):
            arccosh(DualNumber(0.5))
            arccosh(0.5)


    def test_arctanh(self):
        assert isinstance(arctanh(DualNumber(.1, .3)), DualNumber)
        x = DualNumber(.3, .5)
        result = arctanh(x)
        assert np.arctanh(.3) == result.real
        assert pytest.approx(.5/(1 - .3**2)) ==  result.dual

        with pytest.raises(ArithmeticError):
            arctanh(DualNumber(1))
            archtanh(DualNumber(0.5, 0.5))


    def test_sqrt(self):
        assert isinstance(sqrt(DualNumber(4, 2)), DualNumber)
        x = DualNumber(4, -1)
        result = sqrt(x)
        assert np.sqrt(4) == result.real
        assert pytest.approx((-1 * 0.5)*np.power(4, -0.5)) == result.dual

        with pytest.raises(ArithmeticError):
            sqrt(DualNumber(-1))
            sqrt(-1)
