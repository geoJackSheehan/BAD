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
        raise TypeError(fr'{fun} -- Elementary functions can only do computations on DualNumbers, integers, and floats.')

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
    # Ensure the input is as expected, otherwise make minor data cleaning changes
    x = _validate(x, 'exp()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(exp(x.real), x.dual * exp(x.real))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(exp(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
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

    if isinstance(x, DualNumber):
        # Derivative defined (0, infinity); x.real cannot be 0
        if x.real > 0:
            return DualNumber(ln(x.real), x.dual / x.real)

        else:
            raise ArithmeticError('ln() -- Natural log is defined only for values greater than or equal to 1.')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
            f = ReverseMode(ln(x.real))
            x.child.append(f)
            return f

    elif isinstance(x, float):
        # Defined [1, inf)
        if x >= 1:
            return np.log(x)

        else:
            raise ArithmeticError('ln() -- Natural log is defined only for values greater than or equal to 1.')

def logBase(x, base):
    x = _validate(x, 'logBase()')

    # Taking two arguments requires another check not included in basic validation
    if not isinstance(base, (int, float)):
        raise TypeError('logBase() -- Base must be an integer or a float.')

    if isinstance(x, DualNumber):
        # Derivative bounding is the same as the float version, which is delegated below
        return DualNumber(logBase(x.real, base), x.dual / (x.real * ln(base)))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(logBase(x.real, base))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined everywhere that x and base are non-negative
        if x > 0 and base > 0:
            # Use change of base formula with natural base for custom log base computation
            return (np.log(x) / np.log(base))

        else:
            raise ArithmeticError('logBase() -- Ensure base is greater than or equal to 1 and DualNumber real part is greater than 0.')

def sin(x):
    x = _validate(x, 'sin()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(sin(x.real), x.dual * cos(x.real))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(sin(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-inf, inf)
        return np.sin(x)

def cos(x):
    x = _validate(x, 'cos()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(cos(x.real), -1 * sin(x.real) * x.dual)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(cos(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-inf, inf)
        return np.cos(x)

def tan(x):
    x = _validate(x, 'tan()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, 0) U (0, inf), but tan has the same bounding which is handled below before div 0 occurs
        return DualNumber(tan(x.real), x.dual / cos(x.real)**2)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(tan(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined everywhere expect where cosine = 0
        if abs(np.cos(x)) > zero:
            return np.tan(x)

        else:
            raise ArithmeticError('tan() -- Ensure the input does not cause cosine to be 0.')

def csc(x):
    x = _validate(x, 'csc()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(csc(x.real), -1 * x.dual * csc(x.real) * cot(x.real))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(csc(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined everywhere expect where sine = 0
        if abs(np.sin(x)) > zero:
            return (1 / np.sin(x))

        else:
            raise ArithmeticError('csc() -- The sine of the input cannot be 0 due to division.')

def sec(x):
    x = _validate(x, 'sec()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(sec(x.real), sec(x.real) * tan(x.real) * x.dual)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(sec(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined everywhere expect where cosine = 0
        if abs(np.cos(x)) > zero:
            return (1 / np.cos(x))

        else:
            raise ArithmeticError('sec() -- The cosine of the input cannot be 0 due to division.')

def cot(x):
    x = _validate(x, 'cot()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(cot(x.real), -1 * csc(x.real) * csc(x.real) * x.dual)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(cot(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined everywhere expect where tan = 0 (or sine = 0)
        if abs(np.tan(x)) > zero:
            return (1 / np.tan(x))

        else:
            raise ArithmeticError('cot() -- The tangent of the input cannot be 0 due to division.')

def sinh(x):
    x = _validate(x, 'sinh()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(sinh(x.real), cosh(x.real) * x.dual)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(sinh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-infinity, infinity)
        return np.sinh(x)

def cosh(x):
    x = _validate(x, 'cosh()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(cosh(x.real), sinh(x.real) * x.dual)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(cosh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined (-inf, inf)
        return np.cosh(x)

def tanh(x):
    x = _validate(x, 'tanh()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf), Cosh is never 0
        return DualNumber(tanh(x.real), x.dual / cosh(x.real) ** 2)

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(tanh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-inf, inf)
        return np.tanh(x)

def arcsin(x):
    x = _validate(x, 'arcsin()')

    if isinstance(x, DualNumber):
        # Derivative defined (-1, 1)
        if x.real > -1 and x.real < 1:
            return DualNumber(arcsin(x.real), x.dual / sqrt(1 - x.real ** 2))

        else:
            raise ArithmeticError('arcsin() -- Tried to square-root a negative number during dual part creation. Ensure real part is within (-1, 1).')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arcsin(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arcsin(x)

        else:
            raise ArithmeticError('arcsin() -- Arcsine is only defined in the domain [-1, 1]')

def arccos(x):
    x = _validate(x, 'arccos()')

    if isinstance(x, DualNumber):
        # Derivative defined (-1, 1)
        if x.real > -1 and x.real < 1:
            return DualNumber(arccos(x.real), (-1 * x.dual) / sqrt(1 - x.real ** 2))

        else:
            raise ArithmeticError('arccos() -- DualNumber real part must be within defined domain (-1, 1) for dual part creation.')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arccos(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arccos(x)

        else:
            raise ValueError('arccos() -- Function is only defined in the domain [-1, 1]')

def arctan(x):
    x = _validate(x, 'arctan()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(arctan(x.real), x.dual / (1 + x.real ** 2))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arctan(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-inf, inf)
        return np.arctan(x)

def arcsinh(x):
    x = _validate(x, 'arcsinh()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, inf)
        return DualNumber(arcsinh(x.real), x.dual / sqrt(1 + x.real ** 2))

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arcsinh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-inf, inf)
        return np.arcsinh(x)

def arccosh(x):
    x = _validate(x, 'arccosh()')

    if isinstance(x, DualNumber):
        # Derivative defined (1, inf)
        if x.real > 1:
            return DualNumber(arccosh(x.real), x.dual / (sqrt(x.real - 1) * sqrt(x.real + 1)))

        else:
            raise ArithmeticError('arccosh() -- DualNumber real part must be greater than 1 for dual part creation involving square-roots.')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arccosh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for [1, infinity)
        if x >= 1:
            return np.arccosh(x)

        else:
            raise ArithmeticError('arccosh() -- Function is only defined for domain [1, infinity)')

def arctanh(x):
    x = _validate(x, 'arctanh()')

    if isinstance(x, DualNumber):
        # Derivative defined (-inf, -1) U (-1, 1) U (1, inf)
        if x.real is not [-1, 1]:
            return DualNumber(arctanh(x.real), x.dual / (1 - x.real **2))

        else:
            raise ArithmeticError('arctanh() -- DualNumber dual part creation produces divide by 0 if real part is -1 or 1.')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(arctanh(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float):
        # Defined for (-1, 1)
        if x > -1 and x < 1:
            return np.arctanh(x)

        else:
            raise ArithmeticError('arctanh() -- Function is only defined for domain (-1, 1).')

def sqrt(x):
    x = _validate(x, 'sqrt()')

    if isinstance(x, DualNumber):
        # Derivative defined (0, inf)
        if x.real > 0:
            return DualNumber(sqrt(x.real), (x.dual / 2) * (1 / sqrt(x.real)))

        else:
            raise ArithmeticError('sqrt() -- Cannot take the square root of a negative number and cannot divide by 0 for dual part creation.')

    elif isinstance(x, ReverseMode):
        # Same domain as float, no local checking
        f = ReverseMode(sqrt(x.real))
        x.child.append(f)
        return f

    elif isinstance(x, float,):
        # Defined for [0, infinity)
        if x >= 0:
            return np.sqrt(x)

        else:
            raise ArithmeticError('sqrt() -- Cannot take the square root of a negative number.')
