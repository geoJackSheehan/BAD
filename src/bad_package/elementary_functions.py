# Defines and describes the behavior of overloaded operators on different data types within the package
# Numpy handles scalar and vector input by default, returning a scalar float and element-wise vector of floats respectively

import numpy as np
from bad_forward_mode import DualNumber

# OVERLOADING CONSTANTS (/ symbols)
e = np.e
pi = np.pi
inf = np.inf

# OVERLOADING FUNCTIONS
def exp(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\'t gotten her yet')

    elif isinstance(x, (int, float, np.array)):
        # Returns basic e^x computation
        return np.exp(x)

    else:
        raise ValueError('Something went wrong in exp()')

def ln(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')

    elif isinstance(x, (int, float, np.array)):
        # Returns natural log
        if x >= 1:
            return np.log(x)

        else: 
            raise ValueError('Natural log is defined only for values greater than or equal to 1.')

    else:
        raise ValueError('Something went wrong in ln()')

def log10(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')

    elif isinstance(x, (int, float, np.array)):
        # General base in math, but denoted with 10 for user clarity
        return np.log10(x)

    else:
        raise ValueError('Something went wrong in log10()')

def log2(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):
        # Often used by computer scientists for algorithmic efficiency calculations
        return np.log2(x)
    
    else:
        raise ValueError('Something went wrong in log2()')

def logBase(base, x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        # Use change of base formula with common 10 base for custom log base computation
        return (np.log10(x) / np.log10(base))
    
    else:
        raise ValueError('Something went wrong in logBase()')    

def sin(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.sin(x)

    else:
        raise ValueError('Something went wrong in logBase()')           

def cos(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.cos(x)

    else:
        raise ValueError('Something went wrong in cos()')

def tan(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.tan(x)

    else:
        raise ValueError('Something went wrong in tan()')

def csc(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        if np.sin(x) != 0:
            return (1 / np.sin(x))

        else:
            raise ValueError('Cosecant is not defined for the input value.')

    else:
        raise ValueError('Something went wrong in csc()')
            

def sec(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        if np.cos(x) != 0:
            return (1 / np.cos(x))

        else:
            raise ValueError('Secant is not defined for the input value')

    else:
        raise ValueError('Something went wrong in sec()')

def cot(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        if np.tan(x) != 0:
            return (1 / np.tan(x))

        else:
            raise ValueError('Cotangent is not defined for the input value')

    else:
        raise ValueError('Something went wrong in cot()')

def sinh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.sinh(x)

    else:
        raise ValueError('Something went wrong in sinh()')

def cosh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.cosh(x)

    else:
        raise ValueError('Something went wrong in cosh()')

def tanh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.tanh(x)

    else:
        raise ValueError('Something went wrong in tanh()')

def arcsin(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arcsin(x)

    else:
        raise ValueError('Something went wrong in arcsin()')

def arccos(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arccos(x)

    else:
        raise ValueError('Something went wrong in arccos()')

def arctan(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arctan(x)

    else:
        raise ValueError('Something went wrong in arctan()')

def arcsinh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arcsinh(x)

    else:
        raise ValueError('Something went wrong in arcsinh()')

def arccosh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arccosh(x)

    else:
        raise ValueError('Something went wrong in arccosh()')

def arctanh(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.arctanh(x)

    else:
        raise ValueError('Something went wrong in arctan()')

def sqrt(x):
    if isinstance(x, DualNumber):
        raise NotImplementedError('Devs haven\t gotten here yet')
    
    elif isinstance(x, (int, float, np.array)):    
        return np.sqrt(x)

    else:
        raise ValueError('Something went wrong in sqrt()')


if __name__ == "__main__":
    # Basic test code, move to test suite

    print(exp(ln(2)), exp([0, 1, 2, 3]))
    print(cos(1), cos([0, pi/2, pi, 3*pi/2]))
    print(sinh(2*pi), sinh([0, pi/2, pi, 3*pi/2]))
    print(logBase(2, 32))

    # Constants
    print(e, pi, inf)
