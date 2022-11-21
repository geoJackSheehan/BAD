"""
forward module
Implements Forward class: a class to perform the forward mode of automatic differentiation
"""

# Interface that the user instantiates to run Auto-Diff
# User should pass in a list / ndarray of funcs (f1, f2, ...) and varLists (x1, x2, ...) but if they pass in single things we can check for that

# They call AutoDiff, passing a pre-defined function(s) and variables as INTs (coordinate to be evaluated at)
# Ex. AutoDiff(f, (2, 6, 5)) where f = x1 + x2 + x3 and we want the derivative at point x1 = 2, x2 = 6, x3 = 5
# They pass variables IN THE ORDER that the function they defined takes them

# There's probably a bunch of other stuff I'm forgetting but please delete, add, or move anything
# Just wanted to get a shell started so everyone can brainstorm the best way to go about this

# imports
import numpy as np
from bad_package.fad.fad import DualNumber

# from fad import DualNumber

class Forward:
    '''
    Example Driver Script to utilize this interface
    
    ------------------------------------------------
    >>> import numpy as np
    >>> from forward_interface import Forward
    >>> def f(x):
    >>>     return x**2 + (7/x)
    >>> x = np.array([1476])
    >>> forward = Forward(f,x)
    >>> forward.run_forward()
    >>> print(forward.calc_val()); print(forward.calc_jacobian())
    2178576.0047425474
    [2951.999996786892]
    
    '''
    def __init__(self, func, vals):
        if not isinstance(vals, np.ndarray):
            raise TypeError("Second argument must be numpy.ndarray")
        self.func = func
        self.n_vars = len(vals)
        trace = []
        for v in vals:
            trace.append(DualNumber(v,1))
        self.trace = trace
        
    def __repr__(self):
        return f'AutoDiff(functions: {self.funcs}, variables: {self.varList})'
        raise NotImplementedError

    def __str__(self):
        # Are we doing the computational graph here?
        raise NotImplementedError
        
    def run_forward(self):
        #scalar function
        if self.n_vars == 1:
            self.trace[0] = self.func(self.trace[0])
        
        # vector function
        else:
            raise NotImplementedError
            
    def calc_val(self):
        primal = self.trace[0].real
        return primal

    def calc_jacobian(self):
        tangent = [v.dual for v in self.trace]
        return tangent