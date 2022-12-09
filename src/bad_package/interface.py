"""
Explanation
------------------------------------
Forward and backward auto-differentiation user interface

Items
------------------------------------
AutoDiff: 
    Forward-mode implementation. 
    Internally uses DualNumber objects to track accumulated function value and derivative value. 
    Supports any combination of scalar or vector variables and functions. 


"""

import numpy as np
from bad_package.fad import DualNumber
from bad_package.rad import ReverseMode

class AutoDiff:
    '''
    Explanation
    ------------------------------------
    Class used to implement the forward mode of automatic differentiation. 
    Currently supports scalar and vector instances. 

    Attributes
    ------------------------------------
    f:
        List, ndarray, or single function to implement

    var_list:
        List, ndarray, or single number (argument(s)) to evaluate the function at in forward mode

    len_var_list:
        Number of arguments to calculate (dimensionality)

    trace:
        List of DualNumbers to keep track of the current trace of forward mode

    Methods
    ------------------------------------
    __init__(self, f, var_list)
        Instantiate AutoDiff object

    __repr__(self)
        Easy-to-read object instantiation with memory location

    __str__(self)
        Pretty print of the passed function(s) and variable(s)

    _compute(self)
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
        # Flexibility: allow the user to input lists, np.arrays, or single values
        if isinstance(var_list, (list, np.ndarray)):
            var_list = np.array(var_list)
        elif isinstance(var_list, (int, float)):
            var_list = np.array([var_list])
        else:
            raise TypeError(f"Second argument in {print(self)} must be a list or ndarray of integers or float or single integers or floats.")

        if isinstance(f, (list, np.ndarray)):
            f = np.array(f)
        elif callable(f):
            f = np.array([f])
        else:
            raise TypeError(f"First argument in {print(self)} must be a list of ndarray of functions or a single function.")
        
        self.f = f
        self.var_list = var_list
        self.len_var_list = len(var_list)
        self.jacobian = []
        
        trace = []
        for variable in var_list:
            trace.append(DualNumber(float(variable), 1))
        self.trace = trace

        # Automatically starts computation, less steps for the user
        self._compute()

    def __repr__(self):
        '''
        Explanation
        ------------------------------------
        Base print of AutoDiff instantiation with passed values and memory location
        Inputs
        ------------------------------------
        None
        ''' 
        return f'AutoDiff({self.f}, {self.var_list}, id: {id(self)})'

    def __str__(self):
        '''
        Explanation
        ------------------------------------
        Pretty print of AutoDiff instantiation with more information
        Inputs
        ------------------------------------
        None
        ''' 
        return f'f: {self.f}, var_list: {self.var_list}'

    def _compute(self):
        '''
        Explanation
        ------------------------------------
        Calculating primal trace and forward tangent trace to store in self.trace
        Inputs
        ------------------------------------
        None
        '''
        # Iterate through all passed functions (same shape)
        for f in self.f:
            if self.len_var_list == 1:
                # If it's scalar functions, just replace the value at the trace with resulting DualNumber from computation
                self.trace[0] = f(self.trace[0])
                self.jacobian.append(self.trace[0].dual)
            else:
                value = f(self.trace).real
                # Primal trace and tangent trace
                trace, tangent = [], []
                for i in range(self.len_var_list):
                    x = self.trace[i]
                    y = [DualNumber(0, 0)]*self.len_var_list
                    y[i] = x
                    dp = f(y).dual

                    updatedDual = DualNumber(value, dp)
                    trace.append(updatedDual)
                    tangent.append(updatedDual.dual)

                self.jacobian.append(tangent)

    def get_primal(self):
        '''
        Return primal trace of forward mode
        '''
        return self.trace[0].real

    def get_jacobian(self):
        '''
        Return tangent trace of forward mode
        '''
        return self.jacobian

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

# class ReverseAD(AutoDiff):


#     def __init__(self, f, var_list):
#         # Flexibility: allow the user to input lists, np.arrays, or single values
#         if isinstance(var_list, (list, np.ndarray, int, float)):
#             var_list = np.array(var_list)
#         else:
#             raise TypeError(f"Second argument in {print(self)} must be a list or ndarray of integers or float or single integers or floats.")

#         if isinstance(f, (list, np.ndarray)):
#             f = np.array(f)
#         elif callable(f):
#             f = np.array(f)
#         else:
#             raise TypeError(f"First argument in {print(self)} must be a list of ndarray of functions or a single function.")
        
#         self.f = f
#         self.var_list = var_list
#         self.len_var_list = len(var_list)  
        
#         trace = []
#         for variable in var_list:
#             trace.append(ReverseMode(float(variable)))
#         self.trace = trace

#         self._compute()

#     def __repr__(self):
#         '''
#         Explanation
#         ------------------------------------
#         Base print of ReverseAD instantiation with passed values and memory location

#         Inputs
#         ------------------------------------
#         None
#         ''' 
#         return f'ReverseAD(f: {self.f}, var_list: {self.var_list})'

#     def __str__(self):
#         '''
#         Explanation
#         ------------------------------------
#         Pretty print of ReverseAD instantiation with more information

#         Inputs
#         ------------------------------------
#         None
#         ''' 
#         return f'f: {self.f}, var_list: {self.var_list}, variables: {self.var_list}'
        

#     def _compute(self):
#         if self.len_var_list == 1:
#             self.trace[0] = self.f(self.trace[0])
#         else:
#             value = self.f(self.trace).real
#             trace = []
#             for i in range(self.len_var_list):
#                 x = self.trace[i]
#                 y = [ReverseMode(0)]*self.len_var_list
#                 y[i] = x
#                 trace.append(ReverseMode(value))

#             self.trace = trace

#     def get_primal(self):
#         raise NotImplementedError('Reverse AutoDiff does not track primal trace.')

#     def get_jacobian(self):
#         '''
#         Explanation
#         ------------------------------------
#         Return back-propagated chain rule after forward and backwards pass

#         Inputs
#         ------------------------------------
#         None
#         '''
#         return [variable.grad() for variable in self.trace]


# class ReverseAD:

#     def __init__(self, f, var_list):
#         self.f = f
#         self.var_list = var_list
#         self.jacobian = []

#         try:
#             self.len_var_list = len(var_list)
#         except TypeError:
#             self.len_var_list = 1

#         trace = []
#         if self.len_var_list > 1:
#             for variable in var_list:
#                 trace.append(ReverseMode(float(variable)))
#             self.trace = trace
#         self._compute()

#     def _compute(self):
#         # i want to add in a if len = 1 thing, then use this for scalar and another one that appends jacobian list for vector and also i need one for vector variables
#         if self.len_var_list == 1:
#             for i in range(len(self.f)):
#                 x = ReverseMode(self.var_list)
#                 z = self.f[i](x)
#                 z.gradient = 1.0
#                 self.jacobian.append(x.grad())
                
#         elif self.len_var_list > 1:
#             for trace in self.trace:
#                 for i in range(len(self.f)):
#                     z = self.f[i](trace)
#                     z.gradient = 1.0
#                     self.jacobian.append(trace.grad())
#         else:
#             raise TypeError('Variable list cannot be empty!')

        
#     def get_primal(self):
#         raise NotImplementedError('Reverse AutoDiff does not track primal trace.')
        
#     def get_jacobian(self):
#         return self.jacobian

class ReverseAD:

    def __init__(self, f, var_list):
        self.f = f
        self.var_list = var_list
        self.jacobian = []

        try:
            self.len_var_list = len(var_list)
        except TypeError:
            self.len_var_list = 1

        trace = []
        if self.len_var_list > 1:
            for variable in var_list:
                trace.append(ReverseMode(float(variable)))
            self.trace = trace
        self._compute()

    def _compute(self):
        # i want to add in a if len = 1 thing, then use this for scalar and another one that appends jacobian list for vector and also i need one for vector variables
        if self.len_var_list == 1:
            for i in range(len(self.f)):
                x = ReverseMode(self.var_list)
                z = self.f[i](x)
                z.gradient = 1.0
                self.jacobian.append(x.grad())
                
        elif self.len_var_list > 1:
            for i in range(len(self.f)):
                z = self.f[i](self.trace)
                z.gradient = 1.0
            for trace in self.trace:
                self.jacobian.append(trace.grad())
        else:
            raise TypeError('Variable list cannot be empty!')

        
    def get_primal(self):
        raise NotImplementedError('Reverse AutoDiff does not track primal trace.')
        
    def get_jacobian(self):
        return self.jacobian