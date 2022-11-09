import pytest

from bad_package.bad_forward_mode import DualNumber

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
        assert y.dual == 1

    def test_mul(self):
        assert isinstance(DualNumber(1, 1) * 2, DualNumber)
        assert isinstance(2.0 * DualNumber(1, 1), DualNumber)
        assert isinstance(DualNumber(1, 1) * DualNumber(2, 2.0), DualNumber)
        with pytest.raises(TypeError):
            DualNumber(1,1) * '2'
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
