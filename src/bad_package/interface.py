"""
Auto differentiation module
Implements AutoDiff class in order to execute auto differentiation
"""

import numpy as np
from bad_package.fad import DualNumber
from bad_package.rad import ReverseMode

class AutoDiff:
    '''
    Explanation
    ------------------------------------
    Class used to implement the forward mode of automatic differentation
    Currently supports scalar and vector instances

    Attributes
    ------------------------------------
    f:
        Array of functions to implement

    var_list:
        List of numbers (arguments) to calculate forward mode

    len_var_list:
        Number of arguments to calculate

    trace:
        List of DualNumbers to keep track of the current trace of forward mode

    Methods
    ------------------------------------
    __init__(self, f, var_list)
        Instantiate AutoDiff object

    __str__(self)
        Print computational graph -- TODO

    compute(self)
        Calculate forward mode and get primal and tangent trace

    get_primal(self)
        Return primal trace of forward mode

    get_jacobian(self)
        Return tangent trace of forward mode

    get_var_list(self)
        Getter method of self.var_list

    get_f(self)
        Getter method of self.f

    Example Driver Script to utilize forward interface
    --------------------------------------------------
    Scalar:
    
    >>> import numpy as np
    >>> from ad_interface import AutoDiff
    >>> def scalar(x):
    >>>     return 4*x + 3
    >>> x = np.array([2])
    >>> ad = AutoDiff(scalar, x)
    >>> ad.compute()
    >>> print(f'Primal: {ad.get_primal()}')
    Primal: 11
    >>> print(f'Tangent: {ad.get_jacobian()}')
    Tangent: [4]

    Vector:
    
    def vector(x):
        return x[0]**2 + 3*x[1] + 5
    x = np.array([1, 2])
    ad = AutoDiff(vector, x)
    ad.compute()
    print(f'Primal: {ad.get_primal()}')
    >>> 12
    print(f'Tangent: {ad.get_jacobian()}')
    >>> [2, 3]
    '''

    def __init__(self, f, var_list):
        if not isinstance(var_list, np.ndarray):
            raise TypeError("Second argument must be numpy.ndarray")
        self.f = f
        self.var_list = var_list
        self.len_var_list = len(var_list)
        
        trace = []
        for variable in var_list:
            trace.append(DualNumber(float(variable), 1))
        self.trace = trace

    def __str__(self):
        '''
        Pretty print of AutoDiff instantiation with passed values
        '''
        return f'AutoDiff(f: {self.f}, var_list: {self.var_list})'

    def compute(self):
        '''
        Calculating primal trace and forward tangent trace to store in self.trace
        '''
        if self.len_var_list == 1:
            self.trace[0] = self.f(self.trace[0])
        else:
            value = self.f(self.trace).real
            trace = []
            for i in range(self.len_var_list):
                x = self.trace[i]
                y = [DualNumber(0, 0)]*self.len_var_list
                y[i] = x
                dp = self.f(y).dual
                trace.append(DualNumber(value, dp))
            self.trace = trace

    def get_primal(self):
        '''
        Return primal trace of forward mode
        '''
        return self.trace[0].real

    def get_jacobian(self):
        '''
        Return tangent trace of forward mode
        '''
        return [variable.dual for variable in self.trace]

    def get_var_list(self):
        '''
        Return var_list passed by user for forward mode
        '''
        return self.var_list

    def get_f(self):
        '''
        Return function f passed by user for forward mode
        '''
        return self.f

class ReverseAD(AutoDiff):
    # I don't know if we're actually doing this or not, just wanted to add a skeleton

    def __init__(self, f, var_list):
        if not isinstance(var_list, np.ndarray):
            raise TypeError("Second argument must be numpy.ndarray")
        self.f = f
        self.var_list = var_list
        self.len_var_list = len(var_list)  
        
        self.result = self.f(self.var_list)

    def __str__(self):
        '''
        Pretty print of ReverseAD instantiation with passed values
        '''
        return f'ReverseAD(f: {self.f}, var_list: {self.var_list})'

    def get_jacobian(self):
        container = []

        for variable in self.var_list:
            while len(variable.child) > 0:
                variable = variable.child[0]

            container.append(variable)

        return container

        





