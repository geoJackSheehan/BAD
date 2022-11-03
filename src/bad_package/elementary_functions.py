# Defines and describes the behavior of overloaded operators on different data types within the package
# Numpy handles scalar and vector input by default, returning a scalar float and element-wise vector of floats respectively

import numpy as np

# OVERLOADING CONSTANTS (/ symbols)

e = np.e
pi = np.pi
inf = np.inf

# OVERLOADING FUNCTIONS

def exp(x):
    # Returns e^x computation
    return np.exp(x)

def ln(x):
    # Returns natural log
    if x >= 1:
        return np.log(x)

    else: 
        raise ValueError('Natural log is defined only for values greater than or equal to 1.')

def log10(x):
    # General base in math, but denoted with 10 for user clarity
    return np.log10(x)

def log2(x):
    # Often used by computer scientists for algorithmic efficiency calculations
    return np.log2(x)

def logBase(base, x):
    # Use change of base formula with common 10 base for custom log base computation
    return (np.log10(x) / np.log10(base))

def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def tan(x):
    return np.tan(x)

def csc(x):
    if np.sin(x) != 0:
        return (1 / np.sin(x))

    else:
        raise ValueError('Cosecant is not defined for the input value.')

def sec(x):
    if np.cos(x) != 0:
        return (1 / np.cos(x))

    else:
        raise ValueError('Secant is not defined for the input value')

def cot(x):
    if np.tan(x) != 0:
        return (1 / np.tan(x))

    else:
        raise ValueError('Cotangent is not defined for the input value')

def sinh(x):
    return np.sinh(x)

def cosh(x):
    return np.cosh(x)

def tanh(x):
    return np.tanh(x)

def arcsin(x):
    return np.arcsin(x)

def arccos(x):
    return np.arccos(x)

def arctan(x):
    return np.arctan(x)

def arcsinh(x):
    return np.arcsinh(x)

def arccosh(x):
    return np.arccosh(x)

def arctanh(x):
    return np.arctanh(x)

def sqrt(x):
    return np.sqrt(x)


if __name__ == "__main__":
    # Basic test code, move to test suite

    print(exp(ln(2)), exp([0, 1, 2, 3]))
    print(cos(1), cos([0, pi/2, pi, 3*pi/2]))
    print(sinh(2*pi), sinh([0, pi/2, pi, 3*pi/2]))
    print(logBase(2, 32))

    # Constants
    print(e, pi, inf)

    
