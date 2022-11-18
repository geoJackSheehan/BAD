import pytest
from bad_package.fad.fad import DualNumber
import numpy as np

class TestDualNumber:

    def test_init(self):
        assert DualNumber(1, 1).real == 1
        assert DualNumber(1, 1).dual == 1

    def test_add(self):
        assert isinstance(DualNumber(1, 1) + 1, DualNumber)
        assert isinstance(1.0 + DualNumber(1, 1), DualNumber)
        assert isinstance(DualNumber(1, 1) + DualNumber(1, 1), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1, 1) + '1'
            '1' + DualNumber(1, 1)

        x = DualNumber(1, 1)
        y = x + 1
        assert y.real == 2
        assert y.dual == 1

    def test_radd(self):
        x = DualNumber(1, 1)
        y = 1 + x
        assert y.real == 2
        assert y.dual == 1

    def test_sub(self):
        assert isinstance(DualNumber(1, 1) - 1, DualNumber)
        assert isinstance(1.0 - DualNumber(1, 1), DualNumber)
        assert isinstance(DualNumber(1, 1) - DualNumber(1, 1), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1, 1) - '1'
            '1' - DualNumber(1, 1)

        x = DualNumber(1, 1)
        y = x - 1
        assert y.real == 0
        assert y.dual == 1

    def test_rsub(self):
        x = DualNumber(1, 1)
        y = 1 - x
        assert y.real == 0
        assert y.dual == -1

    def test_mul(self):
        assert isinstance(DualNumber(1, 1) * 2, DualNumber)
        assert isinstance(2.0 * DualNumber(1, 1), DualNumber)
        assert isinstance(DualNumber(1, 1) * DualNumber(2, 2.0), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1, 1) * '2'
            '2' * DualNumber(1,1)

        x = DualNumber(1, 1)
        y = x * 2
        assert y.real == 2
        assert y.dual == 2

    def test_rmul(self):
        x = DualNumber(1, 1)
        y = 2 * x
        assert y.real == 2
        assert y.dual == 2

    def test_truediv(self):
        assert isinstance(DualNumber(2, 2) / 2, DualNumber)
        assert isinstance(2 / DualNumber(2, 2), DualNumber)
        assert isinstance(DualNumber(1, 1) / DualNumber(2, 2), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1, 1) / '3'
            '3' / DualNumber(1, 1)

        x = DualNumber(2, 1)
        y = x / 2
        assert y.real == 1
        assert y.dual == 1/2

        z = DualNumber(3, 2)
        dual_dual = x / z
        assert isinstance(dual_dual, DualNumber)
        assert dual_dual.real == 2/3
        assert dual_dual.dual == -1/9

    def test_rtruediv(self):
        x = DualNumber(2, 1)
        y = 2 / x
        assert y.real == 1
        assert y.dual == -1/2

    def test_neg(self):
        assert isinstance(-DualNumber(1, 1), DualNumber)

        x = DualNumber(1, 1)
        y = -x
        assert y.real == -1
        assert y.dual == -1

    def test_pow(self):
        assert isinstance(DualNumber(1, 1) ** 2, DualNumber)
        assert isinstance(DualNumber(1, 1) ** DualNumber(1, 1), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1, 1) ** '5'
            '5' ** DualNumber(1, 1)
        
        x = DualNumber(2, 1)
        y = x**2
        assert y.real == 4
        assert y.dual == 4

    def test_rpow(self):
        x = DualNumber(2, 1)
        y = 2**x
        assert y.real == 4
        assert y.dual == np.log(2)*2 ** 2
