'''
This module defines dunder methods to overload Python built-in operators in the Dual class.
'''

class DualNumber:

    _supported_scalars = (int, float)

    def __init__(self, real, dual=1.0):
        '''
        Explanation
        ------------------------------------
        Constructor for the DualNumber class
        
        Inputs
        ------------------------------------
        real: the value of the object for the user's function
              int or float
        dual: [optional] the derivative of the object for the user's function
              int or float or None
        
        Examples
        ------------------------------------
        >>> x = DualNumber(2)
        >>> x.real
        2
        >>> x.dual
        1
        >>> x = DualNumber(2,2)
        >>> x.real
        2
        >>> x.dual
        2
        
        Notes
        ------------------------------------
        At this stage, DualNumber only supports scalar functions
        '''
        
        self.real = real
        self.dual = dual

        
    def __add__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for addition operator (a + b)
        
        Inputs
        ------------------------------------
        self: object to which the second object is added; a in a + b
              DualNumber object
        other: the object being added; b in a + b
               DualNumber object, int, or float
        
        Outputs
        ------------------------------------
        x = a + b
        a DualNumber object with the value and derivative of the self + other operation
        
        Examples
        ------------------------------------
        Dual + int:
        >>> x = DualNumber(2) + 3
        >>> print(x.real); print(x.dual)
        5
        1.0
        >>> x = DualNumber(2,3) + 3
        >>> print(x.real); print(x.dual)
        5
        3
        
        DualNumber + DualNumber:
        >>> x = DualNumber(2) + DualNumber(3)
        >>> print(x.real); print(x.dual)
        5
        2.0
        >>> x = DualNumber(2,3) + DualNumber(3,3)
        >>> print(x.real); print(x.dual)
        5
        6
        '''
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other+self.real, self.dual)
        else:
            return DualNumber(self.real+other.real, self.dual+other.dual)

    def __radd__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for addition operator in reverse case (b + a)
        
        Inputs
        ------------------------------------
        self: object being added; a in b + a
              DualNumber object, int, or float
        other: the object to which the second object is added; b in b + a
               DualNumber object
        
        Outputs
        ------------------------------------
        x = b + a
        a DualNumber object with the value and derivative of the other + self operation
        
        Examples
        ------------------------------------
        int + DualNumber
        >>> x = 2 + DualNumber(3)
        >>> print(x.real); print(x.dual)
        5
        1.0
        >>> x = 2 + DualNumber(3,10)
        >>> print(x.real); print(x.dual)
        5
        10
        
        DualNumber + DualNumber:
        >>> x = DualNumber(2) + DualNumber(3)
        >>> print(x.real); print(x.dual)
        5
        2.0
        >>> x = DualNumber(2,7) + DualNumber(3,11)
        >>> print(x.real); print(x.dual)
        5
        18
        '''
        
        return self.__add__(other)

    
    def __sub__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for subtraction operator (a - b)
        
        Inputs
        ------------------------------------
        self: object from which the second object is subtracted; a in a - b
              DualNumber object
        other: the object being subtracted; b in a - b
               DualNumber object, int, or float
        
        Outputs
        ------------------------------------
        x = a - b
        a DualNumber object with the value and derivative of the self - other operation
        
        Examples
        ------------------------------------
        DualNumber - int
        >>> x = DualNumber(3) - 2
        >>> print(x.real); print(x.dual)
        1
        5
        >>> x = DualNumber(3,5) - 2
        >>> print(x.real); print(x.dual)
        1
        1.0
        
        DualNumber - DualNumber:
        >>> x = DualNumber(3)-DualNumber(3)
        >>> print(x.real); print(x.dual)
        0
        0.0
        >>> x = DualNumber(3,5)-DualNumber(3,3)
        >>> print(x.real); print(x.dual)
        0
        2
        '''
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real-other, self.dual)
        else:
            return DualNumber(self.real-other.real, self.dual-other.dual)

        
    def __rsub__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for subtraction operator in reverse case (b - a)
        
        Inputs
        ------------------------------------
        self: object being subtracted; a in b - a
              DualNumber object, int, or float
        other: the object from which the second object is subtracted; b in b - a
               DualNumber object
        
        Outputs
        ------------------------------------
        x = b - a
        a DualNumber object with the value and derivative of the other - self operation
        
        Examples
        ------------------------------------
        int - DualNumber
        >>> x = 3 - DualNumber(2)
        >>> print(x.real); print(x.dual)
        1
        -1.0
        >>> x = 3 - DualNumber(2,10)
        >>> print(x.real); print(x.dual)
        1
        -10
        
        DualNumber - DualNumber:
        >>> x = DualNumber(3)-DualNumber(3)
        >>> print(x.real); print(x.dual)
        0
        0.0
        >>> x = DualNumber(3,5)-DualNumber(3,3)
        >>> print(x.real); print(x.dual)
        0
        2
        '''
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other-self.real, -self.dual)
        else:
            return DualNumber(-self.real+other.real, -self.dual+other.dual)
    
    
    def __mul__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for multiplication operator (a * b)
        
        Inputs
        ------------------------------------
        self: first object being multiplied by another; a in a*b
              DualNumber object
        other: object by which the first object is multiplied; b in a*b
               DualNumber object, int, or float
        
        Outputs
        ------------------------------------
        x = a*b
        a DualNumber object with the value and derivative of the self*other operation
        
        Examples
        ------------------------------------
        DualNumber*int
        >>> x = DualNumber(2)*3
        >>> print(x.real); print(x.dual)
        6
        3.0
        >>> x = DualNumber(2,10)*3
        >>> print(x.real); print(x.dual)
        6
        30
        
        DualNumber*DualNumber:
        >>> x = DualNumber(3)*DualNumber(2)
        >>> print(x.real); print(x.dual)
        6
        5.0
        >>> x = DualNumber(3,2)*DualNumber(2,5)
        >>> print(x.real); print(x.dual)
        6
        19
        '''
            
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real*other.real, self.dual*other.real)
        else:
            return DualNumber(self.real*other.real, self.real*other.dual+other.real*self.dual)

        
    def __rmul__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for multiplication operator in reverse case (a*b)
        
        Inputs
        ------------------------------------
        self: object by which the first object is multiplied; a in b*a
              DualNumber object, int, or float
        other: first object being multiplied by another; b in b*a
               DualNumber object
        
        Outputs
        ------------------------------------
        x = b*a
        a DualNumber object with the value and derivative of the other*self operation
        
        Examples
        ------------------------------------
        int*DualNumber:
        >>> x = 3*DualNumber(2)
        >>> print(x.real); print(x.dual)
        6
        3.0
        >>> x = 3*DualNumber(2,10)
        >>> print(x.real); print(x.dual)
        6
        30
        
        DualNumber*DualNumber:
        >>> x = DualNumber(3)*DualNumber(2)
        >>> print(x.real); print(x.dual)
        6
        5.0
        >>> x = DualNumber(3,2)*DualNumber(2,5)
        >>> print(x.real); print(x.dual)
        6
        19
        '''
        
        return self.__mul__(other)

    
    def __truediv__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for division operator (a / b)
        
        Inputs
        ------------------------------------
        self: object being divided; a in a/b
              DualNumber object
        other: object by which the first object is divided; b in a/b
               DualNumber object, int, or float
        
        Outputs
        ------------------------------------
        x = a/b
        a DualNumber object with the value and derivative of the self/other operation
        
        Examples
        ------------------------------------
        DualNumber/int:
        >>> x = DualNumber(6)/3
        >>> print(x.real); print(x.dual)
        2.0
        0.3333333333333333
        >>> x = DualNumber(6,10)/3
        >>> print(x.real); print(x.dual)
        2.0
        3.3333333333333335
        
        DualNumber/DualNumber:
        >>> x = DualNumber(6)/DualNumber(2)
        >>> print(x.real); print(x.dual)
        3.0
        -1.0
        >>> x = DualNumber(10,7)/DualNumber(2,3)
        >>> print(x.real); print(x.dual)
        5.0
        -4.0
        
        Notes
        ------------------------------------
        Only truediv is implemented here (as opposed to truediv and floordiv). Therefore, using the '/' operator will return a floating-point approximation, not the truncated down result of '//'
        '''
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real/other.real, self.dual/other.real)
        else:
            return DualNumber(self.real/other.real, (other.real*self.dual - self.real*other.dual)/(other.real*other.real))

        
    def __rtruediv__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for division operator in reverse case (b / a)
        
        Inputs
        ------------------------------------
        self: object by which the first object is divided; a in b/a
              DualNumber object, int, or float
        other: object being divided; b in b/a
               DualNumber object
        
        Outputs
        ------------------------------------
        x = b/a
        a DualNumber object with the value and derivative of the other/self operation
        
        Examples
        ------------------------------------
        int/DualNumber:
        >>> x = 6/DualNumber(3)
        >>> print(x.real); print(x.dual)
        2.0
        -0.6666666666666666
        >>> x = 10/DualNumber(5,2)
        >>> print(x.real); print(x.dual)
        2.0
        -0.8
        
        DualNumber/DualNumber:
        >>> x = DualNumber(6)/DualNumber(2)
        >>> print(x.real); print(x.dual)
        3.0
        -1.0
        >>> x = DualNumber(10,7)/DualNumber(2,3)
        >>> print(x.real); print(x.dual)
        5.0
        -4.0
        
        Notes
        ------------------------------------
        Only rtruediv is implemented here (as opposed to rtruediv and rfloordiv). Therefore, using the '/' operator will return a floating-point approximation, not the truncated down result of '//'
        '''
        
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other.real/self.real, (-other.real*self.dual)/(self.real*self.real))
        else:
            return DualNumber(other.real/self.real, -(other.real*self.dual - self.real*other.dual)/(other.dual*other.dual))

        
    def __neg__(self):
        if not isinstance(self, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(self, self._supported_scalars):
            return DualNumber(self.real*(-1), self.dual*(-1))
        else:
            return DualNumber(self.real*(-1), self.dual*(-1))

    def __pow__(self, other):
        if not isinstance(self, (*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real ** other, self.dual)
        else:
            raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError


if __name__ == "__main__":
    # rudimentary test code

    import numpy as np
    x1 = 0.5
    z = DualNumber(x1)

    # add
    f1 = z+2
    # radd
    f2 = 2+z
    # sub
    f3 = z-2
    # rsub
    f4 = 2-z
    # mul
    f5 = z*2
    # rmul
    f6 = 2*z
    # truediv
    f7 = z/2
    # rtruediv
    f8 = 2/z
    # neg
    f9 = -z

    f = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9])
    values = ["add", "radd", "sub", "rsub", "mul", "rmul", "truediv", "rtruediv", "neg"]

    for i in range(9):
        print(f' {values[i]}: real = {f[i].real} dual = {f[i].dual}')
