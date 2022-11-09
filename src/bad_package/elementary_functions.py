# Defines and describes the behavior of overloaded operators on different data types within the package
# Numpy handles scalar and vector input by default, returning a scalar float and element-wise vector of floats respectively
# All dual number computations used a trace table with V0 = x.real,  DpV0 = x.dual

import numpy as np
from bad_forward_mode import DualNumber

# OVERLOADING CONSTANTS (/ symbols)
e = np.e
pi = np.pi
inf = np.inf

# OVERLOADING FUNCTIONS
def exp(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementation of exp
        return DualNumber(exp(x.real), x.dual * exp(x.real))

    elif isinstance(x, (int, float, np.array)):
        # Returns basic e^x computation
        return np.exp(x)

    else:
        raise TypeError('The exp() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def ln(x):
    # Using our implementation of ln to handle domain issues
    if isinstance(x, DualNumber):
        if x.real != 0:
            return DualNumber(ln(x.real), x.dual / x.real)
        
        else:
            raise ZeroDivisionError('DualNumber real part cannot be 0 for dual part creation.')

    elif isinstance(x, (int, float, np.array)):
        # Returns natural log
        if x >= 1:
            return np.log(x)

        else: 
            raise ValueError('Natural log is defined only for values greater than or equal to 1.')

    else:
        raise TypeError('The ln() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def logBase(base, x):
    # Used for all other base types; Using our implementation of logBase and ln to handle domain issues
    if isinstance(x, DualNumber):
        if x.real != 0:
            return DualNumber(logBase(base, x.real), x.dual / (x.real * ln(base)))
        
        else:
            raise ZeroDivisionError('logBase() DualNumber real part cannot be 0 for the creation of the dual part.')
    
    elif isinstance(x, (int, float, np.array)):    
        if x > 0 and base > 0:
            # Use change of base formula with common 10 base for custom log base computation
            return (np.log10(x) / np.log10(base))
        
        else:
            raise ValueError('Cannot take log10 of a negative number.')
    
    else:
        raise TypeError('The logBase() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.') 

def sin(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementation of sin and cos
        return DualNumber(sin(x.real), x.dual * cos(x.real))
    
    elif isinstance(x, (int, float, np.array)):    
        return np.sin(x)

    else:
        raise TypeError('The sin() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')      

def cos(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementation of cos and sin
        return DualNumber(cos(x.real), -1 * sin(x.real) * x.dual)

    elif isinstance(x, (int, float, np.array)):    
        return np.cos(x)

    else:
        raise TypeError('The cos() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def tan(x):
    if isinstance(x, DualNumber):
        # Using our implementation of tan and sec to handle domain issues
        return DualNumber(tan(x.real), x.dual * sec(x.real) * sec(x.real))

    elif isinstance(x, (int, float, np.array)):
        if np.cos(x) != 0:    
            return np.tan(x)

        else:
            raise ZeroDivisionError('Tangent is not defined for the input value.')

    else:
        raise TypeError('The tan() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def csc(x):
    if isinstance(x, DualNumber):
        # Using our implementation of csc and cot to handle domain issues
        return DualNumber(csc(x.real), -1 * csc(x.real) * cot(x.dual) * x.dual)
    
    elif isinstance(x, (int, float, np.array)):    
        if np.sin(x) != 0:
            return (1 / np.sin(x))

        else:
            raise ZeroDivisionError('Cosecant is not defined for the input value.')

    else:
        raise TypeError('The csc() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')
            

def sec(x):
    if isinstance(x, DualNumber):
        # Using our implementations of sec and tan to handle domain issues
        return DualNumber(sec(x.real), sec(x.real) * tan(x.real) * x.dual)
    
    elif isinstance(x, (int, float, np.array)):    
        if np.cos(x) != 0:
            return (1 / np.cos(x))

        else:
            raise ZeroDivisionError('Secant is not defined for the input value')

    else:
        raise TypeError('The sec() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def cot(x):
    if isinstance(x, DualNumber):
        # Using our implementation of cot and csc
        return DualNumber(cot(x.real), -1 * csc(x.real) * csc(x.real) * x.dual)

    elif isinstance(x, (int, float, np.array)):    
        if np.tan(x) != 0:
            return (1 / np.tan(x))

        else:
            raise ZeroDivisionError('Cotangent is not defined for the input value.')

    else:
        raise TypeError('The cot() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def sinh(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementations of sinh and cosh
        return DualNumber(sinh(x.real), cosh(x.real) * x.dual)
    
    elif isinstance(x, (int, float, np.array)):    
        return np.sinh(x)

    else:
        raise TypeError('The sinh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def cosh(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementation of cosh and sinh
        return DualNumber(cosh(x.real), sinh(x.real) * x.dual)
    
    elif isinstance(x, (int, float, np.array)):    
        return np.cosh(x)

    else:
        raise TypeError('The cosh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def tanh(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Sech is not directly implemented; Using our tanh and cosh
        return DualNumber(tanh(x.real), x.dual / (1 / cosh(x.real) ** 2))
    
    elif isinstance(x, (int, float, np.array)):    
        return np.tanh(x)

    else:
        raise TypeError('The tanh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def arcsin(x):
    if isinstance(x, DualNumber):
        # Cases in this part deal with dual part creation; Using our implementation of arcsin and sqrt to catch other domain issues
        if x.real > -1 and x.real < 1:
            return DualNumber(arcsin(x.real), x.dual / sqrt(1 - x.real ** 2))

        elif x.real in [-1, 1]:
            raise ZeroDivisionError('arcsin() DualNumber real part cannot be -1 or 1 for dual part creation.')

        else:
            raise ValueError('arcsin() DualNumber real part must be within defined domain (-1, 1) for dual part creation.')
    
    elif isinstance(x, (int, float, np.array)):  
        if x >= -1 and x <= 1:
            return np.arcsin(x)

        else:
            raise ValueError('arcsin() is only defined in the domain [-1, 1]')

    else:
        raise TypeError('The arcsin() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def arccos(x):
    if isinstance(x, DualNumber):
        # Cases in this part deal with dual part creation; Using our implementation of arccos and sqrt to catch other domain issues
        if x.real > -1 and x.real < 1:
            return DualNumber(arccos(x.real), (-1 * x.dual) / sqrt(1 - x.real ** 2))

        elif x.real in [-1, 1]:
            raise ZeroDivisionError('arccos() DualNumber real part cannot be -1 or 1 for dual part creation.')

        else:
            raise ValueError('arccos() DualNumber real part must be within defined domain (-1, 1) for dual part creation.')
    
    elif isinstance(x, (int, float, np.array)):    
        if x >= -1 and x <= 1:
            return np.arccos(x)

        else:
            raise ValueError('arccos() is only defined in the domain [-1, 1]')

    else:
        raise TypeError('The arccos() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def arctan(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Use our implementation of arctan
        return DualNumber(arctan(x.real), x.dual / (1 + x.real ** 2))
    
    elif isinstance(x, (int, float, np.array)):  
        return np.arctan(x)

    else:
        raise TypeError('The arctan() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def arcsinh(x):
    if isinstance(x, DualNumber):
        # Defined for all reals; Use our implementation of arcsinh
        return DualNumber(arcsinh(x.real), x.dual / sqrt(1 + x.real ** 2))
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arcsinh(x)

    else:
        raise TypeError('The arcsinh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def arccosh(x):
    if isinstance(x, DualNumber):
        # Cases involve dual part creation; Using our implementation of arccosh and sqrt to handle domain issues
        if x.real > 1:
            return DualNumber(arccosh(x.real), x.dual / (sqrt(x.real - 1) * sqrt(x.real + 1)))
        
        elif x.real == 1:
            raise ZeroDivisionError('arccosh() DualNumber real part cannot be 1 for dual part creation.')

        else: 
            raise ValueError('arccosh() DualNumber real part must be within defined domain (1, infinity) for dual part creation.')
    
    elif isinstance(x, (int, float, np.array)):    
        if x >= 1:
            return np.arccosh(x)
        
        else:
            raise ValueError('arccosh() is only defined for domain [1, infinity)')

    else:
        raise TypeError('The arccosh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')
        

def arctanh(x):
    if isinstance(x, DualNumber):
        # Cases involve dual part creation; Using our implementation of arctanh to handle domain issues
        if x.real is not [-1, 1]:
            return DualNumber(arctanh(x.real), x.dual / (1 - x.real **2))
        
        else:
            raise ZeroDivisionError('arctanh(): DualNumber real part cannot be -1 or 1 for dual part creation.')
    
    elif isinstance(x, (int, float, np.array)):  
        if x > -1 and x < 1:  
            return np.arctanh(x)
        
        else: 
            raise ValueError('arctanh() is only defined for domain (-1, 1)')

    else:
        raise TypeError('The arctanh() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')

def sqrt(x):
    # Using our implementation of sqrt to catch domain issues
    if isinstance(x, DualNumber):
        if x.real > 0:
            return DualNumber(sqrt(x.real), (x.dual / 2) * (1 / sqrt(x.real)))
        
        else:
            raise ValueError('Cannot take the square root of a negative number or 0 for dual part creation.')
    
    elif isinstance(x, (int, float, np.array)):    
        if x >= 0:
            return np.sqrt(x)

        else:
            raise ValueError('Cannot take the square root of a negative number.')

    else:
        raise TypeError('The sqrt() function can only handle single DualNumbers, ints, floats, or lists, numpy arrays of these types.')


if __name__ == "__main__":
    # Basic test code, move to test suite

    print(exp(ln(2)), exp([0, 1, 2, 3]))
    print(cos(1), cos([0, pi/2, pi, 3*pi/2]))
    print(sinh(2*pi), sinh([0, pi/2, pi, 3*pi/2]))
    print(logBase(2, 32))

    # Constants
    print(e, pi, inf)
