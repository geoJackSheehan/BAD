import pytest
from bad_package.rad import ReverseMode
import numpy as np

class TestReverseMode:

    def test_init(self):
        rm = ReverseMode(3)
        
        assert rm.real == 3
        assert len(rm.child) == 0


    def test_add(self):
        rm = ReverseMode(49)
        rm2 = ReverseMode(1)

        # Supports int, floats, RM objects
        res1 = rm + 6
        assert len(rm.child) == 1
        res2 = rm + 3.14
        assert len(rm.child) == 2
        res3 = rm + rm2
        assert len(rm.child) == 3
        assert len(rm2.child) == 1

        # Int add test and trace tracking
        assert res1.real == rm.real + 6
        assert res1.real == 6 + rm.real
        assert len(res1.child) == 1
        
        # Float add test and trace tracking
        assert res2.real == rm.real + 3.14
        assert res2.real == 3.14 + rm.real
        assert len(res2.child) == 1

        # Adding two RM objects
        assert res3.real == rm.real + rm2.real
        assert res3.real == rm2.real + rm.real
        assert len(res3.child) == 1

        # Throws errors
        with pytest.raises(TypeError):
            ReverseMode('7') + 4
            ReverseMode([1, 2, 3]) + 6
            ReverseMode(np.NaN) + 1
            ReverseMode((1, 2)) + 0

    def test_radd(self):
        rm = ReverseMode(0)
        rm2 = ReverseMode(1)

        res1 = 1 + rm
        assert len(rm.child) == 1
        res2 = -4 + rm
        assert len(rm.child) == 2
        res3 = 9.99 + rm2

        # Int radd test and trace check
        assert res1.real == 1 + rm.real
        assert res1.real == rm.real + 1
        assert len(res1.child) == 1

        # Negative int test radd and trace check 
        assert res2.real == rm.real - 4
        assert res2.real == -4 + rm.real
        assert len(res2.child) == 1

        # Float radd test and trace check
        assert res3.real == rm2.real +9.99
        assert res3.real == 9.99 + rm2.real
        assert len(res3.child) == 1

        # Throws errors
        with pytest.raises(TypeError):
            3 + ReverseMode('7')
            6 + ReverseMode([1, 2, 3])
            1 + ReverseMode(np.NaN)
            0 + ReverseMode((1, 2))


    def test_sub(self):
        rm = ReverseMode(1)
        rm2 = ReverseMode(-10)

        res1 = rm - 10
        assert len(rm.child) == 1
        res2 = rm - 2.718
        assert len(rm.child) == 2
        res3 = rm2 - 10
        assert len(rm.child) == 3
        res4 = rm - rm2
        assert len(rm.child) == 4
        assert len(rm2.child) == 1
        
        # Int subtract test and trace check
        assert res1.real == rm.real - 10
        assert res1.real == -10 + rm.real
        assert len(res1.child) == 1

        # Float subtract test and trace check
        assert res2.real == rm.real - 2.718
        assert res2.real == -2.718 + rm.real
        assert len(res2.child) == 1

        # Negative reverse mode subtract test and trace check
        assert res3.real == rm2.real - 10
        assert res3.real == -10 + rm2.real
        assert len(res3.child) == 1

        # Both ReverseMode subtract test and trace check
        assert res4.real == rm.real - rm2.real
        assert res4.real == -rm2.real + rm.real
        assert len(res4.child) == 1
        
        # Throws errors
        with pytest.raises(TypeError):
            ReverseMode('7') - ReverseMode('4')
            ReverseMode([1, 2, 3]) - ReverseMode(2)
            ReverseMode(np.NaN) - ReverseMode(np.Inf)
            ReverseMode((1, 2)) - ReverseMode(2)


    def test_rsub(self):
        rm = ReverseMode(6)
        rm2 = ReverseMode(-5)

        res1 = 10 - rm
        res2 = 2.718 - rm
        res3 = -10 - rm2
        
        # Int reverse subtract test
        assert res1.real == 10 - rm.real
        assert res1.real ==  -rm.real + 10

        # Float reverse subtract test
        assert res2.real == 2.718 - rm.real
        assert res2.real == -rm.real + 2.718

        # Negative int reverse subtraction
        assert res3.real == -10 + -rm2.real
        assert res3.real == rm2.real - 10
        
        # Throws errors
        with pytest.raises(TypeError):
            20 - ReverseMode('7')
            4 - ReverseMode([1, 2, 3])
            -10 - ReverseMode(np.Inf)
            1 - ReverseMode((1, 2))
