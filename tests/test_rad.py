import pytest
from bad_package.rad import ReverseMode
from bad_package.fad import DualNumber
import numpy as np

class TestReverseMode:

    def test_init(self):
        rm = ReverseMode(3)
        
        assert rm.real == 3
        assert len(rm.child) == 0
        assert rm.gradient == None

        with pytest.raises(TypeError):
            ReverseMode('a')
            ReverseMode([1,2,3])
            ReverseMode((6, 7))
            ReverseMode(DualNumber(1))
            ReverseMode([])

#     def test_grad(self):
#         rm = ReverseMode(3)
#         res1 = rm**2
#         res1.gradient = 1.0
#         assert rm.grad() == sum(weight * var.grad() for weight, var in rm.child)
#         assert rm.grad() == 6
        
    def test_grad(self):
        rm = ReverseMode(3)
        res1 = rm**2
        res1.gradient = 1.0
        assert rm.grad() == sum(dvj_dvi * df_dvj.grad() for dvj_dvi, df_dvj in rm.child)
        assert rm.grad() == 6
    
    def test_child(self):
        # Testing the self.child container
        rm = ReverseMode(3)
        rm2 = ReverseMode(1)

        res1 = ((rm * 3) + 1) - rm2
        assert len(rm.child) == 3
        assert len(rm2.child) == 1

        res2 = (1 / rm) + 6
        assert len(rm.child) == 5

    def test_return(self):
        rm = ReverseMode(2)
        rm2 = ReverseMode(-5)

        res1 = rm + 4 - rm2 * 3
        assert isinstance(res1, ReverseMode)


    def test_add(self):
        rm = ReverseMode(49)
        rm2 = ReverseMode(1)

        # Supports int, floats, RM objects
        res1 = rm + 6
        res2 = rm + 3.14
        res3 = rm + rm2

        # Int add test
        assert res1.real == rm.real + 6
        assert res1.real == 6 + rm.real
        
        # Float add test
        assert res2.real == rm.real + 3.14
        assert res2.real == 3.14 + rm.real

        # Adding two RM objects
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
        res2 = -4 + rm
        res3 = 9.99 + rm2

        # Int radd test
        assert res1.real == 1 + rm.real
        assert res1.real == rm.real + 1

        # Negative int test radd 
        assert res2.real == rm.real - 4
        assert res2.real == -4 + rm.real

        # Float radd test
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
        res2 = rm - 2.718
        res3 = rm2 - 10
        res4 = rm - rm2
        
        # Int subtract test 
        assert res1.real == rm.real - 10
        assert res1.real == -10 + rm.real

        # Float subtract test 
        assert res2.real == rm.real - 2.718
        assert res2.real == -2.718 + rm.real
 
        # Negative reverse mode subtract test 
        assert res3.real == rm2.real - 10
        assert res3.real == -10 + rm2.real

        # Both ReverseMode subtract test
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
        res2 = 2.718 - rm
        res3 = -10 + rm2
        
        # Int reverse subtract test
        assert res1.real == 10 - rm.real
        assert res1.real == (-1 * rm.real) + 10

        # Float reverse subtract test
        assert res2.real == 2.718 - rm.real
        assert res2.real == -rm.real + 2.718

        # Negative int reverse subtraction
        assert res3.real == -10 + rm2.real
        assert res3.real == rm2.real - 10
        
        # Throws errors
        with pytest.raises(TypeError):
            20 - ReverseMode('7')
            4 - ReverseMode([1, 2, 3])
            -10 - ReverseMode(np.Inf)
            1 - ReverseMode((1, 2))


    def test_mul(self):
        rm = ReverseMode(1)
        rm2 = ReverseMode(-2)

        res1 = rm * rm2
        res2 = rm * 4
        res3 = rm * -1

        # Checking RM object mul is commutative
        assert res1.real == rm.real * rm2.real
        assert res1.real == rm2.real * rm.real

        # Checking int mul commutative
        assert res2.real == rm.real * 4
        assert res2.real == 4 * rm.real

        # Checking neg int mul commutative
        assert res3.real == rm.real * -1
        assert res3.real == 0 - rm.real

        # Validating type inputs
        with pytest.raises(TypeError):
            ReverseMode('2') * 3
            ReverseMode((1,2)) * 4
            rm * '4'
            rm * DualNumber(1)


    def test_rmul(self):
        rm = ReverseMode(3)

        res1 = -3 * rm
        res2 = 3.14159 * rm
        res3 = -1 * rm * 2

        # Checking neg int mul commutative
        assert res1.real == rm.real * -3
        assert res1.real == -3 * rm.real

        # Checking float mul commutative
        assert res2.real == rm.real * 3.14159
        assert res2.real == 3.14159 * rm.real

        # Checking neg int (multiple ops) commutative
        assert res3.real == -2 * rm.real
        assert res3.real == rm.real * -2

        # Validating type inputs
        with pytest.raises(TypeError):
            3 * ReverseMode('2')
            4 * ReverseMode([1,2])
            '4' * rm
            DualNumber(1) * rm


    def test_truediv(self):
        rm = ReverseMode(2)
        rm2 = ReverseMode(3)

        res1 = rm/2
        res2 = rm/-1
        res3 = rm/9.18
        res4 = rm/rm2

        # Non-commutative, checking ints, neg ints, floats, and both RM objects
        assert res1.real == rm.real / 2
        assert res2.real == -1 * rm.real
        assert res3.real == rm.real / 9.18
        assert res4.real == rm.real / rm2.real

        # Validation input
        with pytest.raises(TypeError):
            rm / '2'
            rm / [1, 2, 3]
            rm / (2, 3)
            rm / DualNumber(1)

        # Making sure can't div by 0 with a 0 RM object
        with pytest.raises(ArithmeticError):
            rm / ReverseMode(0)


    def test_rtruediv(self):
        rm = ReverseMode(2)
        rm2 = ReverseMode(3)

        res1 = 2/rm
        res2 = -1/rm
        res3 = 9.18/rm
        res4 = 0/rm

        # Non-commutative, checking ints, neg ints, floats, and 0 numerator
        assert res1.real == 2 / rm.real
        assert res2.real == -1 / rm.real
        assert res3.real == 9.18 / rm.real
        assert res4.real == 0

        # Validating input types
        with pytest.raises(TypeError):
            '2' / rm2
            [1, 2, 3] / rm
            (2, 3) / rm2
            DualNumber(1) / rm


    def test_neg(self):
        rm = ReverseMode(3)
        res1 = -rm

        assert res1.real == -1 * rm.real
        assert res1.real == -rm.real

    def test_pow(self):
        rm = ReverseMode(6)

        res1 = rm ** 2
        res2 = rm ** 3.2

        assert res1.real == rm.real ** 2
        assert res1.real == rm.real * rm.real
        assert res2.real == rm.real ** 3.2

        with pytest.raises(TypeError):
            rm ** 'a'
            rm ** [1,2,3]
            rm ** (1,2)
            rm ** DualNumber(2)


    def test_rpow(self):
        rm = ReverseMode(4)

        res1 = 2 ** rm
        res2 = 1.08 ** rm

        assert res1.real == 2 ** rm.real
        assert res1.real == 2 * 2 * 2 * 2
        assert res2.real == 1.08 ** rm.real
        assert res2.real == 1.08 * 1.08 * 1.08 * 1.08

        with pytest.raises(TypeError):
            'a' ** rm
            [1,2,3] ** rm
            (1,2) ** rm
            DualNumber(2) ** rm
