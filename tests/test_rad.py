import pytest
from bad_package.rad import ReverseMode
import numpy as np

class TestReverseMode:

    def test_init(self):
        rm = ReverseMode(3)
        
        assert rm.real == 3
        assert len(rm.child) == 0
        assert rm.grad == None


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

        # Int add test
        assert isinstance(res1, ReverseMode)
        assert res1.real == rm.real + 6
        assert res1.real == 6 + rm.real
        
        # Float add test
        assert isinstance(res2, ReverseMode)
        assert res2.real == rm.real + 3.14
        assert res2.real == 3.14 + rm.real

        # Adding two RM objects
        assert isinstance(res3, ReverseMode)
        assert res3.real == rm.real + rm2.real
        assert res3.real == rm2.real + rm.real

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

        # Int radd test
        assert isinstance(res1, ReverseMode)
        assert res1.real == 1 + rm.real
        assert res1.real == rm.real + 1

        # Negative int test radd 
        assert isinstance(res2, ReverseMode)
        assert res2.real == rm.real - 4
        assert res2.real == -4 + rm.real

        # Float radd test
        assert isinstance(res3, ReverseMode)
        assert res3.real == rm2.real +9.99
        assert res3.real == 9.99 + rm2.real

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
        assert len(rm2.child) == 1
        res4 = rm - rm2
        assert len(rm.child) == 4
        assert len(rm2.child) == 2
        
        # Int subtract test 
        assert isinstance(res1, ReverseMode)
        assert res1.real == rm.real - 10
        assert res1.real == -10 + rm.real

        # Float subtract test 
        assert isinstance(res2, ReverseMode)
        assert res2.real == rm.real - 2.718
        assert res2.real == -2.718 + rm.real
 
        # Negative reverse mode subtract test 
        assert isinstance(res3, ReverseMode)
        assert res3.real == rm2.real - 10
        assert res3.real == -10 + rm2.real

        # Both ReverseMode subtract test
        assert isinstance(res4, ReverseMode)
        assert res4.real == rm.real - rm2.real
        assert res4.real == -rm2.real + rm.real
        
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
        assert len(rm.child) == 1
        res2 = 2.718 - rm
        assert len(rm.child) == 2
        res3 = -10 - rm2
        assert len(rm2.child) == 1
        
        # Int reverse subtract test
        assert isinstance(res1, ReverseMode)
        assert res1.real == 10 - rm.real
        assert res1.real ==  -rm.real + 10

        # Float reverse subtract test
        assert isinstance(res2, ReverseMode)
        assert res2.real == 2.718 - rm.real
        assert res2.real == -rm.real + 2.718

        # Negative int reverse subtraction
        assert isinstance(res3, ReverseMode)
        assert res3.real == -10 + -rm2.real
        assert res3.real == rm2.real - 10
        
        # Throws errors
        with pytest.raises(TypeError):
            20 - ReverseMode('7')
            4 - ReverseMode([1, 2, 3])
            -10 - ReverseMode(np.Inf)
            1 - ReverseMode((1, 2))
