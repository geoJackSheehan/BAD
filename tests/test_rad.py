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
        res3 = -4 + rm
        assert len(rm.child) == 3
        res4 = rm + rm2
        assert len(rm.child) == 4
        assert len(rm2.child) == 1
        assert isinstance((res1, res2, res3, res4), (ReverseMode, ReverseMode, ReverseMode, ReverseMode))

        # Int add test and trace tracking
        assert res1.real == rm.real + 6
        assert res1.real == 6 + rm.real
        assert len(res1.child) == 1
        
        # Float add test and trace tracking
        assert res2.real == rm.real + 3.14
        assert res2.real == 3.14 + rm.real
        assert len(res2.child) == 1

        # Negative int test adding and trace tracking 
        assert res3.real == rm.real - 4
        assert res3.real == -4 + rm.real
        assert len(res3.child) == 1

        # Adding two RM objects
        assert res4.real == rm.real + rm2.real
        assert res4.real == rm2.real + rm.real

        # Throws errors
        with pytest.raises(TypeError):
            ReverseMode('7')
            ReverseMode([1, 2, 3])
            ReverseMode(np.NaN)
            ReverseMode((1, 2))
            ReverseMode(np.Inf)

    def test_radd(self):
        raise NotImplementedError

    def test_sub(self):
        raise NotImplementedError

    def test_rsub(self):
        raise NotImplementedError

