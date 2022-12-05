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
        # DualNumber
        assert isinstance(tan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 5)
        result = tan(x)
        assert np.tan(2) == result.real
        assert pytest.approx(5/(np.cos(2)**2)) == result.dual

        x_neg = DualNumber(-0.5, -3)
        result_neg = tan(x_neg)
        assert np.tan(-0.5) == result_neg.real
        assert pytest.approx(-3/(np.cos(-0.5)**2)) == result_neg.dual

        # ReverseMode
        assert isinstance(tan(ReverseMode(2)), ReverseMode)
        x = ReverseMode(2)
        result = tan(x)
        assert np.tan(2) == result.real

        x_neg = ReverseMode(-0.5)
        result_neg = tan(x_neg)
        assert np.tan(-0.5) == result_neg.real

        # General
        with pytest.raises(ArithmeticError):
            tan(DualNumber(pi/2))
            tan(ReverseMode(pi/2))


    def test_csc(self):
        # DualNumber
        # csc'(x) = -csc(x)cot(x)
        assert isinstance(csc(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = csc(x)
        assert 1/sin(2) == result.real
        assert pytest.approx(-3*(1/np.sin(2))*(1/np.tan(2))) == result.dual

        # ReverseMode
        assert isinstance(csc(ReverseMode(2)), ReverseMode)
        x = ReverseMode(2)
        result = csc(x)
        assert 1/sin(2) == result.real

        # General
        with pytest.raises(ArithmeticError):
            csc(DualNumber(pi))
            csc(ReverseMode(pi))


    def test_sec(self):
        # DualNumber
        # sec'(x) = sec(x)tan(x)
        assert isinstance(sec(DualNumber(2, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = sec(x)
        assert 1/cos(2) == result.real
        assert pytest.approx(3*(1/np.cos(2))*np.tan(2)) == result.dual

        # ReverseMode
        assert isinstance(sec(ReverseMode(2)), ReverseMode)
        x = ReverseMode(2)
        result = sec(x)
        assert 1/cos(2) == result.real

        # General
        with pytest.raises(ArithmeticError):
            sec(DualNumber(pi/2))
            sec(ReverseMode(pi/2))


    def test_cot(self):
        # DualNumber
        # cot'(x) = -csc^2(x)
        assert isinstance(cot(DualNumber(2, 2)), DualNumber)
        x = DualNumber(4, 3)
        result = cot(x)
        assert 1/tan(4) == result.real
        assert pytest.approx(-3*((1/np.sin(4))**2)) == result.dual

        # ReverseMode
        assert isinstance(cot(ReverseMode(2)), ReverseMode)
        x = ReverseMode(4)
        result = cot(x)
        assert 1/tan(4) == result.real

        # General
        with pytest.raises(ArithmeticError):
            cot(DualNumber(pi))
            cot(ReverseMode(pi))


    def test_sinh(self):
        # DualNumber
        assert isinstance(sinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(-0.25, 1.5)
        result = sinh(x)
        assert np.sinh(-0.25) == result.real
        assert pytest.approx(1.5*np.cosh(-0.25)) == result.dual

        # ReverseMode
        assert isinstance(sinh(ReverseMode(1)), ReverseMode)
        x = ReverseMode(-0.25)
        result = sinh(x)
        assert np.sinh(-0.25) == result.real


    def test_cosh(self):
        # DualNumber
        assert isinstance(cosh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 5)
        result = cosh(x)
        assert np.cosh(2) == result.real
        assert pytest.approx(5*np.sinh(2)) == result.dual

        # ReverseMode
        assert isinstance(cosh(ReverseMode(1)), ReverseMode)
        x = ReverseMode(2)
        result = cosh(x)
        assert np.cosh(2) == result.real


    def test_tanh(self):
        # DualNumber
        # tanh'(x) = 1 - tanh^2(x)
        assert isinstance(tanh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(.1, .2)
        result = tanh(x)
        assert pytest.approx(np.tanh(.1)) == result.real
        assert pytest.approx(.2 * (1/(np.cosh(.1)))**2) == result.dual

        # ReverseMode
        assert isinstance(tanh(ReverseMode(2)), ReverseMode)
        x = DualNumber(.1)
        result = tanh(x)
        assert pytest.approx(np.tanh(.1)) == result.real


    def test_arcsin(self):
        # DualNumber
        assert isinstance(arcsin(DualNumber(0.9, 1)), DualNumber)
        x = DualNumber(0.25, 5)
        result = arcsin(x)
        assert np.arcsin(0.25) == result.real
        assert pytest.approx(5/np.sqrt(1 - 0.25**2)) ==  result.dual

        # ReverseMode
        assert isinstance(arcsin(ReverseMode(0.9)), ReverseMode)
        x = ReverseMode(0.25)
        result = arcsin(x)
        assert np.arcsin(0.25) == result.real    

        # General
        with pytest.raises(ArithmeticError):
            arcsin(DualNumber(-1, -1))
            arcsin(ReverseMode(-2))
            arcsin(1.1)


    def test_arccos(self):
        # DualNumber
        assert isinstance(arccos(DualNumber(0.9, 3)), DualNumber)
        x = DualNumber(0.75, -.2)
        result = arccos(x)
        assert np.arccos(0.75) == result.real
        assert pytest.approx((-1)*(-.2)/(np.sqrt((1-(0.75)**2)))) == result.dual

        # ReverseMode
        assert isinstance(arccos(ReverseMode(0.9)), ReverseMode)
        x = DualNumber(0.75)
        result = arccos(x)
        assert np.arccos(0.75) == result.real

        # General
        with pytest.raises(ArithmeticError):
            arccos(DualNumber(-1, -1))
            arccos(ReverseMode(1.2))
            arccos(-1.1)


    def test_arctan(self):
        # DualNumber
        assert isinstance(arctan(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 3)
        result = arctan(x)
        assert np.arctan(2) == result.real
        assert pytest.approx(3*(1/(1+(2**2)))) == result.dual

        # ReverseMode
        assert isinstance(arctan(ReverseMode(2)), ReverseMode)
        x = ReverseMode(2)
        result = arctan(x)
        assert np.arctan(2) == result.real


    def test_arcsinh(self):
        # DualNumber
        assert isinstance(arcsinh(DualNumber(1, 1)), DualNumber)
        x = DualNumber(2, 3)
        result = arcsinh(x)
        assert np.arcsinh(2) == result.real
        assert pytest.approx(3/(np.sqrt(2**2 + 1))) == result.dual

        # ReverseMode
        assert isinstance(arcsinh(ReverseMode(1)), ReverseMode)
        x = ReverseMode(2)
        result = arcsinh(x)
        assert np.arcsinh(2) == result.real


    def test_arccosh(self):
        # DualNumber
        assert isinstance(arccosh(DualNumber(2, 2)), DualNumber)
        x = DualNumber(2, 0.3)
        result = arccosh(x)
        assert np.arccosh(2) == result.real
        assert pytest.approx(0.3/(np.sqrt(2**2 - 1))) == result.dual

        # ReverseMode
        assert isinstance(arccosh(ReverseMode(2)), ReverseMode)
        x = ReverseMode(4)
        result = arccosh(x)
        assert np.arccosh(4) == result.real

        # General
        with pytest.raises(ArithmeticError):
            arccosh(DualNumber(0.5))
            arccosh(0.5)
            arccos(ReverseMode(-10))


    def test_arctanh(self):
        # DualNumber
        assert isinstance(arctanh(DualNumber(0.1, 0.3)), DualNumber)
        x = DualNumber(0.3, 0.5)
        result = arctanh(x)
        assert np.arctanh(0.3) == result.real
        assert pytest.approx(0.5/(1 - 0.3**2)) ==  result.dual

        # ReverseMode
        assert isinstance(arctanh(ReverseMode(0.1)), ReverseMode)
        x = ReverseMode(0.3)
        result = arctanh(x)
        assert np.arctanh(0.3) == result.real 

        # General
        with pytest.raises(ArithmeticError):
            arctanh(DualNumber(1))
            arctanh(DualNumber(0.5, 0.5))
            arctanh(ReverseMode(-1))


    def test_sqrt(self):
        # DualNumber
        assert isinstance(sqrt(DualNumber(4, 2)), DualNumber)
        x = DualNumber(4, -1)
        result = sqrt(x)
        assert np.sqrt(4) == result.real
        assert pytest.approx((-1 * 0.5)*np.power(4, -0.5)) == result.dual

        # ReverseMode
        assert isinstance(sqrt(ReverseMode(4)), ReverseMode)
        x = ReverseMode(4)
        result = sqrt(x)
        assert np.sqrt(4) == result.real

        # General
        with pytest.raises(ArithmeticError):
            sqrt(DualNumber(-1))
            sqrt(-1)
            sqrt(ReverseMode(-0.5))
