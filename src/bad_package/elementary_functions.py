# Defines and describes the behavior of overloaded operators on different data types within the package
# All dual number computations used a trace table with V0 = x.real,  DpV0 = x.dual
# Elementary functions are recursive in the DualNumber computation s.t. we pass the real part and/or dual part to another function 
#   within this module to compute the scalar output. Ensures all Numpy errors are overwritten with custom errors so the user does not
#   need to interact with numpy using our module (expect for in cases where they are not using an AD object). Also allows us to 
#   ensure functional domain restrictions are met.
# Domain restrictions on functions are imposed by most restrictive case 

# Expectations:
# 1. Any elementary function is being applied to a single DualNumber instance or int / float
# 2. DualNumber real parts have been converted to floats already, the dual part will be done by us explicitly (setting the seed vector, ect)
# 3. NotImplementedError raised if there is (1) unexpected behavior or (2) devs haven't gotten to implementing something yet
# 4. TypeError raised if an inappropriate data type was passed from the user
# 5. ArithmeticError raised if there was a (generally) mathematically inappropriate calculation about to happen 

import numpy as np
from bad_forward_mode import DualNumber

# OVERLOADING CONSTANTS (/ symbols)
e = np.e
pi = np.pi
inf = np.inf

# Helper functions
def _validate(x, fun):
    # So we avoid any kind of truncation errors and things, better to do so explicitly
    if isinstance(x, int):
        return float(x)

    # Check if the element is something we can do the computation with (would have casted int to float already)
    if not isinstance(x, (DualNumber, float)):
        raise TypeError(f'{fun} -- Elementary functions can only do computations on DualNumbers, integers, and floats.')

# OVERLOADING FUNCTIONS
def exp(x):
    # Ensure the input is as expected, otherwise make minor data cleaning changes
    x = _validate(x, 'exp()')

    if isinstance(x, DualNumber):
        # Defined for all reals; Using our implementation of exp
        return DualNumber(exp(x.real), x.dual * exp(x.real))

    elif isinstance(x, float):
        # Returns basic e^x computation
        return np.exp(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def ln(x):
    x = _validate(x, 'ln()')

    if isinstance(x, DualNumber):
        # ln Defined for [1, infinity); x.real cannot be 0
        if x.real >= 1:
            return DualNumber(ln(x.real), x.dual / x.real)
        
        else:
            raise ArithmeticError('ln() -- Natural log is defined only for values greater than or equal to 1.')

    elif isinstance(x, float):
        # Returns natural log of scalar input to user or DualNumber creation
        if x >= 1:
            return np.log(x)

        else: 
            raise ArithmeticError('ln() -- Natural log is defined only for values greater than or equal to 1.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def logBase(base, x):
    x = _validate(x, 'logBase()')

    # Taking two arguments requires another check not included in basic validation
    if not isinstance(base, (int, float)):
        raise TypeError('logBase() -- Base must be an integer or a float.')

    if isinstance(x, DualNumber):
        # log (general) is defined for (0, infinity); ln handled elsewhere; x.real != 0
        if x.real > 0 and base > 0:
            # ln is used in the base because it is the easiest version of custom log base derivative
            return DualNumber(logBase(base, x.real), x.dual / (x.real * ln(base)))
        
        else:
            raise ArithmeticError('logBase() -- Ensure base is greater than or equal to 1 and DualNumber real part is greater than 0.')
    
    elif isinstance(x, float):    
        # Defined everywhere that x and base are non-negative
        if x > 0 and base > 0:
            # Use change of base formula with common 10 base for custom log base computation
            return (np.log10(x) / np.log10(base))
        
        else:
            raise ArithmeticError('logBase() -- Ensure base is greater than or equal to 1 and DualNumber real part is greater than 0.')
    
    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.') 

def sin(x):
    x = _validate(x, 'sin()')

    if isinstance(x, DualNumber):
        return DualNumber(sin(x.real), x.dual * cos(x.real))
    
    elif isinstance(x, float):
        # Defined for (-infinity, infinity)
        return np.sin(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')      

def cos(x):
    x = _validate(x, 'cos()')

    if isinstance(x, DualNumber):
        return DualNumber(cos(x.real), -1 * sin(x.real) * x.dual)

    elif isinstance(x, float):    
        # Defined for (-infinity, infinity)
        return np.cos(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def tan(x):
    x = _validate(x, 'tan()')

    if isinstance(x, DualNumber):
        return DualNumber(tan(x.real), x.dual * sec(x.real) * sec(x.real))

    elif isinstance(x, float):
        # Defined everywhere expect where cosine = 0
        if np.cos(x) != 0:    
            return np.tan(x)

        else:
            raise ArithmeticError('tan() -- Ensure the input is defined within tangent\'s domain.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def csc(x):
    x = _validate(x, 'csc()')

    if isinstance(x, DualNumber):
        return DualNumber(csc(x.real), -1 * csc(x.real) * cot(x.dual) * x.dual)
    
    elif isinstance(x, float):
        # Defined everywhere expect where sine = 0     
        if np.sin(x) != 0:
            return (1 / np.sin(x))

        else:
            raise ArithmeticError('csc() -- The sine of the input cannot be 0 due to division.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')
            

def sec(x):
    x = _validate(x, 'sec()')

    if isinstance(x, DualNumber):
        return DualNumber(sec(x.real), sec(x.real) * tan(x.real) * x.dual)
    
    elif isinstance(x, float):    
        if np.cos(x) != 0:
            return (1 / np.cos(x))

        else:
            raise ArithmeticError('sec() -- The cosine of the input cannot be 0 due to division.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def cot(x):
    x = _validate(x, 'cot()')

    if isinstance(x, DualNumber):
        return DualNumber(cot(x.real), -1 * csc(x.real) * csc(x.real) * x.dual)

    elif isinstance(x, float):    
        if np.tan(x) != 0:
            return (1 / np.tan(x))

        else:
            raise ArithmeticError('cot() -- The tangent of the input cannot be 0 due to division.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def sinh(x):
    x = _validate(x, 'sinh()')

    if isinstance(x, DualNumber):
        return DualNumber(sinh(x.real), cosh(x.real) * x.dual)
    
    elif isinstance(x, float):  
        # Defined for (-infinity, infinity)  
        return np.sinh(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def cosh(x):
    x = _validate(x, 'cosh()')

    if isinstance(x, DualNumber):
        # Defined for (-infinity, infinity)
        return DualNumber(cosh(x.real), sinh(x.real) * x.dual)
    
    elif isinstance(x, float):    
        return np.cosh(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def tanh(x):
    x = _validate(x, 'tanh()')

    if isinstance(x, DualNumber):
        if cosh(x.real) != 0:
            return DualNumber(tanh(x.real), x.dual / (1 / cosh(x.real) ** 2))

        else:
            raise ArithmeticError('tanh() -- DualNumber dual part creation tried to divide by zero since cosh of real = 0.')
    
    elif isinstance(x, float):    
        # Defined for (-infinity, infinity)
        return np.tanh(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def arcsin(x):
    x = _validate(x, 'arcsin()')

    if isinstance(x, DualNumber):
        # Cases in this part deal with dual part creation
        if x.real > -1 and x.real < 1:
            return DualNumber(arcsin(x.real), x.dual / sqrt(1 - x.real ** 2))

        else:
            raise ArithmeticError('arcsin() -- Tried to square-root a negative number during dual part creation. Ensure real part is within (-1, 1).')
    
    elif isinstance(x, float):  
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arcsin(x)

        else:
            raise ArithmeticError('arcsin() -- Arcsine is only defined in the domain [-1, 1]')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def arccos(x):
    x = _validate(x, 'arccos()')

    if isinstance(x, DualNumber):
        # Cases in this part deal with dual part creation
        if x.real > -1 and x.real < 1:
            return DualNumber(arccos(x.real), (-1 * x.dual) / sqrt(1 - x.real ** 2))

        else:
            raise ArithmeticError('arccos() -- DualNumber real part must be within defined domain (-1, 1) for dual part creation.')
    
    elif isinstance(x, float):    
        # Defined for [-1, 1]
        if x >= -1 and x <= 1:
            return np.arccos(x)

        else:
            raise ValueError('arccos() -- Function is only defined in the domain [-1, 1]')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def arctan(x):
    x = _validate(x, 'arctan()')

    if isinstance(x, DualNumber):
        # Defined for all reals
        return DualNumber(arctan(x.real), x.dual / (1 + x.real ** 2))
    
    elif isinstance(x, float):  
        # Defined for all reals
        return np.arctan(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def arcsinh(x):
    x = _validate(x, 'arcsinh()')

    if isinstance(x, DualNumber):
        # Defined for all reals
        return DualNumber(arcsinh(x.real), x.dual / sqrt(1 + x.real ** 2))
    
    elif isinstance(x, float):    
        # Defined for all reals
        return np.arcsinh(x)

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def arccosh(x):
    x = _validate(x, 'arccosh()')

    if isinstance(x, DualNumber):
        # Cases involve dual part creation
        if x.real > 1:
            return DualNumber(arccosh(x.real), x.dual / (sqrt(x.real - 1) * sqrt(x.real + 1)))
        
        else: 
            raise ArithmeticError('arccosh() -- DualNumber real part must be greater than 1 for dual part creation involving square-roots.')
    
    elif isinstance(x, float):    
        # Defined for [1, infinity)
        if x >= 1:
            return np.arccosh(x)
        
        else:
            raise ArithmeticError('arccosh() -- Function is only defined for domain [1, infinity)')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')
        

def arctanh(x):
    x = _validate(x, 'arctanh()')

    if isinstance(x, DualNumber):
        # Cases involve dual part creation
        if x.real is not [-1, 1]:
            return DualNumber(arctanh(x.real), x.dual / (1 - x.real **2))
        
        else:
            raise ArithmeticError('arctanh() -- DualNumber dual part creation produces divide by 0 if real part is -1 or 1.')
    
    elif isinstance(x, float):  
        # Defined for (-1, 1)
        if x > -1 and x < 1:  
            return np.arctanh(x)
        
        else: 
            raise ArithmeticError('arctanh() -- Function is only defined for domain (-1, 1).')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

def sqrt(x):
    x = _validate(x, 'sqrt()')

    if isinstance(x, DualNumber):
        # Domain for reals mut be positive, or divide by 0 error or negative square-root
        if x.real > 0:
            return DualNumber(sqrt(x.real), (x.dual / 2) * (1 / sqrt(x.real)))
        
        else:
            raise ArithmeticError('sqrt() -- Cannot take the square root of a negative number and cannot divide by 0 for dual part creation.')
    
    elif isinstance(x, float,):    
        # Defined for [0, infinity)
        if x >= 0:
            return np.sqrt(x)

        else:
            raise ArithmeticError('sqrt() -- Cannot take the square root of a negative number.')

    else:
        raise NotImplementedError('Unexpected behavior detected. Please contact developers.')

if __name__ == "__main__":
    # Basic test code, move to test suite

    print(exp(2))
    # print(exp(1))

    # print(exp(['3']))
    # print(exp(ln(2)), exp([0, 1, 2, 3]))
    # print(cos(1), cos([0, pi/2, pi, 3*pi/2]))
    # print(sinh(2*pi), sinh([0, pi/2, pi, 3*pi/2]))
    # print(logBase(2, 32))

    # # Constants
    # print(e, pi, inf)
