# Defines and describes the behavior of overloaded operators on different data types within the package
import numpy as np
from bad_package.fad import DualNumber
from bad_package.rad import ReverseMode

__all__ = ['e', 'pi', 'zero', 'exp', 'ln', 'logBase', 'sin', 'cos', 'tan', 'csc', 'sec', 'cot', 'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh', 'sqrt']

# OVERLOADING CONSTANTS
e = np.e
pi = np.pi

# A machine precision 0 that Numpy produces
zero = np.sin(pi)

# Helper functions
def _validate(x, fun):
    '''
    Explanation
    ------------------------------------
    Private method for validating user input

    Inputs
    ------------------------------------
    x: the object that's about to have an elementary function applied to it
    fun: (str) the elementary function calling this one

    Outputs
    ------------------------------------
    if x is an integer, return float
    if x is a float, return float
    if x is a DualNumber, return DualNumber
    if x is a ReverseMode, return ReverseMode

    Raises
    ------------------------------------
    TypeError: invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    # So we avoid any kind of truncation errors and things, better to do so explicitly
    if isinstance(x, int):
        return float(x)
    # Check if the element is something we can do the computation with (would have casted int to float already)
    elif isinstance(x, (DualNumber, ReverseMode, float)):
        return x
    else:
        raise TypeError(f'{fun} -- Elementary functions can only do computations on DualNumbers, ReverseModes, integers, and floats')

# OVERLOADING FUNCTIONS
def exp(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy exp function

    Inputs
    ------------------------------------
    x: the object we want to raise e to

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(exp(x.real), exp(x) derivative)
    if x is a ReverseMode, return ReverseMode(exp(x.real))
    if x is a float, return exp(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'exp()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(exp(x.real), x.dual * exp(x.real))
        else:
            f = ReverseMode(exp(x.real))
            x.child.append((exp(x.real), f))
            return f
    else:
        # Defined (-inf, inf)
        return np.exp(x)

def ln(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy log function

    Inputs
    ------------------------------------
    x: the object we want to natural log

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(ln(x.real), ln(x) derivative)
    if x is a ReverseMode, return ReverseMode(ln(x.real))
    if x is a float, return ln(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: functional domain error (asymptotes / generally undefined)
    '''
    x = _validate(x, 'ln()')

    if not isinstance(x, float):
        # Derivative cannot have x.real == 0
        if x.real > 0:
            if isinstance(x, DualNumber):
                return DualNumber(ln(x.real), x.dual / x.real)
            else:
                f = ReverseMode(ln(x.real))
                x.child.append(((1/x.real), f))
                return f
        else:
            raise ArithmeticError(f'ln({type(x)}) -- Natural log is defined only for values greater than or equal to 1')
    else:
        # Defined [1, inf)
        if x >= 1:
            return np.log(x)
        else:
            raise ArithmeticError(f'ln({type(x)}) -- Natural log is defined only for values greater than or equal to 1')

def logBase(x, base):
    '''
    Explanation
    ------------------------------------
    Overloading change of base for numpy log

    Inputs
    ------------------------------------
    x: the object we want to take the log of
    base: the base of log we want

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(logBase(x.real, base), logBase(x, base) derivative)
    if x is a ReverseMode, return ReverseMode(logBase(x.real, base))
    if x is a float, return ln(x)/ln(base) = log_{base}(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    TypeError: invalid base type, must be int or float
    ArithmeticError: functional domain error (undefined log values and the denominator cannot be 0)
    '''
    x = _validate(x, 'logBase()')

    # Taking two arguments requires another check not included in basic validation
    if not isinstance(base, (int, float)):
        raise TypeError(f'logBase({type(x)}, {base}) -- Base must be an integer or a float.')

    if not isinstance(x, float):
        # Defined everywhere that x and base are non-negative
        if x.real > 0 and base > 0:
            if isinstance(x, DualNumber):
                return DualNumber(logBase(x.real, base), x.dual / (x.real * ln(base)))
            else:
                f = ReverseMode(logBase(x.real, base))
                x.child.append((1/(x.real*ln(base)), f))
                return f
        else:
            raise ArithmeticError(f'logBase({type(x)}) -- Ensure base is greater than or equal to 1 and real part is greater than 0')
    else:
        # Defined everywhere that x and base are non-negative
        if x > 0 and base > 0:
            return (np.log(x) / np.log(base))
        else:
            raise ArithmeticError(f'logBase({type(x)}) -- Ensure base is greater than or equal to 1 and real part is greater than 0')

def sin(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy sin function

    Inputs
    ------------------------------------
    x: the object we want to take the sin function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(sin(x.real), sin(x) derivative)
    if x is a ReverseMode, return ReverseMode(sin(x.real))
    if x is a float, return sin(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'sin()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(sin(x.real), x.dual * cos(x.real))
        else:
            f = ReverseMode(sin(x.real))
            x.child.append((cos(x.real), f))
            return f
    else:
        # Defined for (-inf, inf)
        return np.sin(x)

def cos(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy cosine function

    Inputs
    ------------------------------------
    x: the object we want to take the cosine function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(cos(x.real), cos(x) derivative)
    if x is a ReverseMode, return ReverseMode(cos(x.real))
    if x is a float, return cos(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'cos()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(cos(x.real), -1 * sin(x.real) * x.dual)
        else:
            f = ReverseMode(cos(x.real))
            x.child.append((-1 * sin(x.real), f))
            return f
    else:
        # Defined for (-inf, inf)
        return np.cos(x)

def tan(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy tangent function

    Inputs
    ------------------------------------
    x: the object we want to take the tangent function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(tan(x.real), tan(x) derivative)
    if x is a ReverseMode, return ReverseMode(tan(x.real))
    if x is a float, return tan(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid x, cos(x) cannot be 0.
    '''
    x = _validate(x, 'tan()')

    if not isinstance(x, float):
        # Derivative defined (-inf, 0) U (0, inf), but tan has the same bounding which is handled below before div 0 occurs
        if isinstance(x, DualNumber):
            return DualNumber(tan(x.real), x.dual / cos(x.real)**2)
        else:
            f = ReverseMode(tan(x.real))
            x.child.append((1/(cos(x.real)**2), f))
            return f
    else:
        # Defined everywhere expect where cosine = 0
        if abs(np.cos(x)) > zero:
            return np.tan(x)
        else:
            raise ArithmeticError(f'tan({type(x)}) -- Ensure the input does not cause cosine to be 0')

def csc(x):
    '''
    Explanation
    ------------------------------------
    Method to calculate the cosecant.
    Note that: csc(x) = 1/sin(x).

    Inputs
    ------------------------------------
    x: the object we want to take the csc function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(csc(x.real), csc(x) derivative)
    if x is a ReverseMode, return ReverseMode(csc(x.real))
    if x is a float, return csc(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid x, sin(x) cannot be 0
    '''
    x = _validate(x, 'csc()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf), other div zero issues (csc and cot) handled in their functions
        if isinstance(x, DualNumber):
            return DualNumber(csc(x.real), -1 * x.dual * csc(x.real) * cot(x.real))
        else:
            f = ReverseMode(csc(x.real))
            x.child.append((-csc(x.real)*(1/tan(x.real)), f))
            return f
    else:
        # Defined everywhere expect where sine = 0
        if abs(np.sin(x)) > zero:
            return (1 / np.sin(x))
        else:
            raise ArithmeticError(f'csc({type(x)}) -- The sine of the input cannot be 0 due to division')

def sec(x):
    '''
    Explanation
    ------------------------------------
    Method to calculate the secant.
    Note that: sec(x) = 1/cos(x).

    Inputs
    ------------------------------------
    x: the object we want to take the sec function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(sec(x.real), sec(x) derivative)
    if x is a ReverseMode, return ReverseMode(sec(x.real))
    if x is a float, return sec(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid x, cos(x) cannot be 0
    '''
    x = _validate(x, 'sec()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf), other div zero issues (sec) handled below
        if isinstance(x, DualNumber):
            return DualNumber(sec(x.real), sec(x.real) * tan(x.real) * x.dual)
        else:
            f = ReverseMode(sec(x.real))
            x.child.append((sec(x.real)*tan(x.real), f))
            return f
    else:
        # Defined everywhere expect where cosine = 0
        if abs(np.cos(x)) > zero:
            return (1 / np.cos(x))
        else:
            raise ArithmeticError(f'sec({type(x)}) -- The cosine of the input cannot be 0 due to division')

def cot(x):
    '''
    Explanation
    ------------------------------------
    Method to calculate the cotangent.
    Note that: cot(x) = 1/tan(x) = cos(x)/sin(x).

    Inputs
    ------------------------------------
    x: the object we want to take the cot function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(cot(x.real), cot(x) derivative)
    if x is a ReverseMode, return ReverseMode(cot(x.real))
    if x is a float, return cot(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid x, tan(x) cannot be 0
    '''
    x = _validate(x, 'cot()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf), other div zero issues (csc) handled in its function
        if isinstance(x, DualNumber):
            return DualNumber(cot(x.real), -1 * csc(x.real) * csc(x.real) * x.dual)
        else:
            f = ReverseMode(cot(x.real))
            x.child.append(((-csc(x.real))**2, f))
            return f
    else:
        # Defined everywhere expect where tan = 0 (or sine = 0)
        if abs(np.tan(x)) > zero:
            return (1 / np.tan(x))
        else:
            raise ArithmeticError(f'cot({type(x)}) -- The tangent of the input cannot be 0 due to division')

def sinh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy sinh function

    Inputs
    ------------------------------------
    x: the object we want to take the sinh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(sinh(x.real), sinh(x) derivative)
    if x is a ReverseMode, return ReverseMode(sinh(x.real))
    if x is a float, return sinh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'sinh()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(sinh(x.real), cosh(x.real) * x.dual)
        else:
            f = ReverseMode(sinh(x.real))
            x.child.append((cosh(x.real), f))
            return f
    else:
        # Derivative defined (-inf, inf)
        return np.sinh(x)

def cosh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy cosh function

    Inputs
    ------------------------------------
    x: the object we want to take the cosh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(cosh(x.real), cosh(x) derivative)
    if x is a ReverseMode, return ReverseMode(cosh(x.real))
    if x is a float, return cosh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'cosh()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(cosh(x.real), sinh(x.real) * x.dual)
        else:
            f = ReverseMode(cosh(x.real))
            x.child.append((sinh(x.real), f))
            return f
    else:
        # Defined (-inf, inf)
        return np.cosh(x)

def tanh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy tanh function

    Inputs
    ------------------------------------
    x: the object we want to take the tanh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(tanh(x.real), tanh(x) derivative)
    if x is a ReverseMode, return ReverseMode(tanh(x.real))
    if x is a float, return tanh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'tanh()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf), Cosh is never 0
        if isinstance(x, DualNumber):
            return DualNumber(tanh(x.real), x.dual / cosh(x.real) ** 2)
        else:
            f = ReverseMode(tanh(x.real))
            x.child.append(((1/cosh(x.real))**2 ,f))
            return f
    else:
        # Defined for (-inf, inf)
        return np.tanh(x)

def arcsin(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arcsin function

    Inputs
    ------------------------------------
    x: the object we want to take the arcsin function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arcsin(x.real), arcsin(x) derivative)
    if x is a ReverseMode, return ReverseMode(arcsin(x.real))
    if x is a float, return arcsin(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid real part of DualNumber, must be within (-1, 1)
    ArithmeticError: invalid x, arcsin() is only defined for the domain [-1, 1]
    '''
    x = _validate(x, 'arcsin()')

    if not isinstance(x, float):
        # Derivative defined (-1, 1)
        if x.real > -1 and x.real < 1:
            if isinstance(x, DualNumber):
                return DualNumber(arcsin(x.real), x.dual / sqrt(1 - x.real ** 2))
            else:
                f = ReverseMode(arcsin(x.real))
                x.child.append((1/sqrt(1 - (x.real)**2), f))
                return f
        else:
            raise ArithmeticError(f'arcsin({type(x)}) -- Tried to square-root a negative number. Ensure real part is within (-1, 1)')
    else:
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arcsin(x)
        else:
            raise ArithmeticError(f'arcsin({type(x)}) -- Arcsine is only defined in the domain [-1, 1]')

def arccos(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arccos function

    Inputs
    ------------------------------------
    x: the object we want to take the arccos function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arccos(x.real), arccos(x) derivative)
    if x is a ReverseMode, return ReverseMode(arccos(x.real))
    if x is a float, return arccos(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: real part of DualNumber is only defined for the domain (-1, 1)
    ArithmeticError: invalid x, arccos() is only defined for the domain [-1, 1]
    '''
    x = _validate(x, 'arccos()')

    if not isinstance(x, float):
        # Derivative defined (-1, 1)
        if x.real > -1 and x.real < 1:
            if isinstance(x, DualNumber):
                return DualNumber(arccos(x.real), (-1 * x.dual) / sqrt(1 - x.real ** 2))
            else:
                f = ReverseMode(arccos(x.real))
                x.child.append((-1/sqrt(1-(x.real)**2), f))
                return f  
        else:
            raise ArithmeticError(f'arccos({type(x)}) --  Real part must be within defined domain (-1, 1) for partial derivatives')
    else:
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arccos(x)
        else:
            raise ArithmeticError(f'arccos({type(x)}) -- Function is only defined in the domain [-1, 1]')

def arctan(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arctan function

    Inputs
    ------------------------------------
    x: the object we want to take the arctan function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arctan(x.real), arctan(x) derivative)
    if x is a ReverseMode, return ReverseMode(arctan(x.real))
    if x is a float, return arctan(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'arctan()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(arctan(x.real), x.dual / (1 + x.real ** 2))
        else:
            f = ReverseMode(arctan(x.real))
            x.child.append((1/(1 + x.real**2), f))
            return f
    else:
        # Defined for (-inf, inf)
        return np.arctan(x)

def arcsinh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arcsinh function

    Inputs
    ------------------------------------
    x: the object we want to take the arcsinh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arcsinh(x.real), arcsinh(x) derivative)
    if x is a ReverseMode, return ReverseMode(arcsinh(x.real))
    if x is a float, return arcsinh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    '''
    x = _validate(x, 'arcsinh()')

    if not isinstance(x, float):
        # Derivative defined (-inf, inf)
        if isinstance(x, DualNumber):
            return DualNumber(arcsinh(x.real), x.dual / sqrt(1 + x.real ** 2))
        else:
            f = ReverseMode(arcsinh(x.real))
            x.child.append((1 / sqrt(1 + x.real ** 2), f))
            return f
    else:
        # Defined for (-inf, inf)
        return np.arcsinh(x)

def arccosh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arccosh function

    Inputs
    ------------------------------------
    x: the object we want to take the arccosh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arccosh(x.real), arccosh(x) derivative)
    if x is a ReverseMode, return ReverseMode(arccosh(x.real))
    if x is a float, return arccosh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid real part for DualNumber, must be greater than 1
    ArithmeticError: invalid x, only defined for the domain [1, inf)
    '''
    x = _validate(x, 'arccosh()')

    if not isinstance(x, float):
        # Derivative defined (1, inf)
        if x.real > 1:
            if isinstance(x, DualNumber):
                return DualNumber(arccosh(x.real), x.dual / (sqrt(x.real - 1) * sqrt(x.real + 1)))
            else:
                f = ReverseMode(arccosh(x.real))
                x.child.append((1 / (sqrt(x.real - 1) * sqrt(x.real + 1)), f))
                return f
        else:
            raise ArithmeticError(f'arccosh({type(x)}) -- Real part must be greater than 1 for derivative involving square-roots')
    else:
        # Defined for [1, infinity)
        if x >= 1:
            return np.arccosh(x)
        else:
            raise ArithmeticError(f'arccosh({type(x)}) -- Function is only defined for domain [1, infinity)')

def arctanh(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy arctanh function

    Inputs
    ------------------------------------
    x: the object we want to take the arctanh function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(arctanh(x.real), arctanh(x) derivative)
    if x is a ReverseMode, return ReverseMode(arctanh(x.real))
    if x is a float, return arctanh(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: invalid real part for DualNumber, must not be -1 or 1
    ArithmeticError: invalid x, only defined for domain (-1, 1)
    '''
    x = _validate(x, 'arctanh()')

    if not isinstance(x, float):
        # Derivative defined (-inf, -1) U (-1, 1) U (1, inf)
        if x.real is not [-1, 1]:
            if isinstance(x, DualNumber):
                return DualNumber(arctanh(x.real), x.dual / (1 - x.real **2))
            else:
                f = ReverseMode(arctanh(x.real))
                x.child.append((1 / (1 - x.real ** 2), f))
                return f
        else:
            raise ArithmeticError(f'arctanh({type(x)}) --  Derivative calculation produces divide by 0 if real part is -1 or 1')
    else:
        # Defined for (-1, 1)
        if x > -1 and x < 1:
            return np.arctanh(x)
        else:
            raise ArithmeticError(f'arctanh({type(x)}) -- Function is only defined for domain (-1, 1)')

def sqrt(x):
    '''
    Explanation
    ------------------------------------
    Overloading numpy sqrt function

    Inputs
    ------------------------------------
    x: the object we want to take the square root function of

    Outputs
    ------------------------------------
    if x is a DualNumber, return DualNumber(sqrt(x.real), sqrt(x) derivative)
    if x is a ReverseMode, return ReverseMode(sqrt(x.real))
    if x is a float, return sqrt(x)

    Raises
    ------------------------------------
    TypeError: (outsourced) invalid x type, must be int, float, DualNumber, or ReverseMode
    ArithmeticError: real part of DualNumber must be greater than 0
    ArithmeticError: invalid x, cannot be negative
    '''
    x = _validate(x, 'sqrt()')

    if not isinstance(x, float):
        # Derivative defined (0, inf)
        if x.real > 0:
            if isinstance(x, DualNumber):
                return DualNumber(sqrt(x.real), (x.dual / 2) * (1 / sqrt(x.real)))
            else:
                f = ReverseMode(sqrt(x.real))
                x.child.append((0.5* x.real ** (-0.5), f))
                return f
        else:
            raise ArithmeticError(f'sqrt({type(x)}) -- Derivative cannot take the square root of a negative number and cannot divide by 0')
    else:
        # Defined for [0, infinity)
        if x >= 0:
            return np.sqrt(x)
        else:
            raise ArithmeticError(f'sqrt({type(x)}) -- Cannot take the square root of a negative number')