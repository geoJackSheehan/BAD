'''
This module defines dunder methods to overload Python built-in operators for Reverse Mode.
'''
# Imports
import numpy as np

# Reverse Class
class ReverseMode():
    
    _supported_scalars = (int, float)

    def __init__(self, real):
        '''
        Explanation
        ------------------------------------
        Constructor for the ReverseMode class
        
        Inputs
        ------------------------------------
        real: the value of the object for the user's function
              int or float
              
        Outputs
        ------------------------------------
        self: ReverseMode object
            self.real: value of the object
            self.child: stores derivatives and relationship to object for children of ojbect
            self.gradient: derivative calculation in reverse mode
        
        Examples
        ------------------------------------
        >>> x = ReverseMode(3)
        >>> x.real
        3
        >>> x.child
        []
        >>> x.gradient
        
        Notes
        ------------------------------------
        At this stage, ReverseMode only supports scalar functions.
        '''
        
        self.real = real
        self.child = []
        self.gradient = None
        
    def grad(self):
        '''
        Explanation
        ------------------------------------
        Function to recursively calculate the derivative with respect to self. Recreates the computation graph. 
        
        Inputs
        ------------------------------------
        self: ReverseMode object
        
        Outputs
        ------------------------------------
        self.der: the derivative of the function with respect to the current variable
                  float
        
        Examples
        ------------------------------------
        >>> rm = ReverseMode(3)
        >>> res1 = rm**2
        >>> rm.gradient = 1.0
        >>> rm.grad()
        6
        
        Notes
        ------------------------------------
        Do we have to assign gradient to 1 before calling this in order for it to work? 
        Uses sum() for situations when self has more than one child
        '''
        
        if self.gradient is None:
            self.gradient = sum(dvj_dvi * df_dvj.grad() for dvj_dvi, df_dvj in self.child)
        return self.gradient

    def __add__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for addition operator (a + b)
        
        Inputs
        ------------------------------------
        self: object to which the second object is added; a in a + b
              ReverseMode object
        other: the object being added; b in a + b
               ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = a + b
        a ReverseMode object with the value of the self + other operation
        
        Examples
        ------------------------------------
        ReverseMode + int:
        >>> x = ReverseMode(2) + 3
        >>> print(x.real)
        5
        
        ReverseMode + ReverseMode:
        >>> x = ReverseMode(2) + ReverseMode(3)
        >>> print(x.real)
        5
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real + other)
            self.child.append((1.0, f))
        else:
            f = ReverseMode(self.real + other.real)
            other.child.append((1.0, f))
            self.child.append((1.0, f))
        return f

    def __radd__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for addition operator in reverse case (b + a)
        
        Inputs
        ------------------------------------
        self: object being added; a in b + a
              ReverseMode object, int, or float
        other: the object to which the second object is added; b in b + a
               ReverseMode object
        
        Outputs
        ------------------------------------
        x = b + a
        a ReverseMode object with the value of the other + self operation
        
        Examples
        ------------------------------------
        int + ReverseMode
        >>> x = 2 + ReverseMode(3)
        >>> print(x.real)
        5
        
        ReverseMode + ReverseMode:
        >>> x = ReverseMode(2) + ReverseMode(3)
        >>> print(x.real)
        5
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
              ReverseMode object
        other: the object being subtracted; b in a - b
               ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = a - b
        a ReverseMode object with the value of the self - other operation
        
        Examples
        ------------------------------------
        ReverseMode - int
        >>> x = ReverseMode(3) - 2
        >>> print(x.real)
        1
        
        ReverseMode - ReverseMode:
        >>> x = ReverseMode(3)-ReverseMode(3)
        >>> print(x.real)
        0
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real - other)
            self.child.append((1.0, f))
        else:
            f = ReverseMode(self.real - other.real)
            other.child.append((1.0, f))
            self.child.append((-1.0, f))
        return f

    def __rsub__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for subtraction operator in reverse case (b - a)
        
        Inputs
        ------------------------------------
        self: object being subtracted; a in b - a
              ReverseMode object, int, or float
        other: the object from which the second object is subtracted; b in b - a
               ReverseMode object
        
        Outputs
        ------------------------------------
        x = b - a
        a ReverseMode object with the value of the other - self operation
        
        Examples
        ------------------------------------
        int - ReverseMode
        >>> x = 3 - ReverseMode(2)
        >>> print(x.real)
        1
        
        ReverseMode - ReverseMode:
        >>> x = ReverseMode(3)-ReverseMode(3)
        >>> print(x.real)
        0
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError("Type not supported: must be int or float")
            
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(other - self.real)
            self.child.append((1.0, f))
        else:
            f = ReverseMode(-self.real + other.real)
            other.child.append((1.0, f))
            self.child.append((-1.0, f))
        return f
    
    def __mul__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for multiplication operator (a * b)
        
        Inputs
        ------------------------------------
        self: first object being multiplied by another; a in a*b
              ReverseMode object
        other: object by which the first object is multiplied; b in a*b
               ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = a*b
        a ReverseMode object with the value of the self*other operation
        
        Examples
        ------------------------------------
        ReverseMode*int
        >>> x = ReverseMode(2)*3
        >>> print(x.real)
        6
        
        ReverseMode*ReverseMode:
        >>> x = ReverseMode(3)*ReverseMode(2)
        >>> print(x.real); print(x.dual)
        6
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real * other)
            self.child.append((other, f))
        else:
            f = ReverseMode(self.real * other.real)
            other.child.append((self.real, f))
            self.child.append((other.real, f))
        return f

    def __rmul__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for multiplication operator in reverse case (a*b)
        
        Inputs
        ------------------------------------
        self: object by which the first object is multiplied; a in b*a
              ReverseMode object, int, or float
        other: first object being multiplied by another; b in b*a
               ReverseMode object
        
        Outputs
        ------------------------------------
        x = b*a
        a ReverseMode object with the value of the other*self operation
        
        Examples
        ------------------------------------
        int*ReverseMode:
        >>> x = 3*ReverseMode(2)
        >>> print(x.real)
        6
        
        ReverseMode*ReverseMode:
        >>> x = ReverseMode(3)*ReverseMode(2)
        >>> print(x.real)
        6
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
              ReverseMode object
        other: object by which the first object is divided; b in a/b
               ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = a/b
        a ReverseMode object with the value of the self/other operation
        
        Examples
        ------------------------------------
        ReverseMode/int:
        >>> x = ReverseMode(6)/3
        >>> print(x.real)
        2.0
        
        ReverseMode/ReverseMode:
        >>> x = ReverseMode(6)/ReverseMode(2)
        >>> print(x.real)
        3.0
        
        Notes
        ------------------------------------
        Only truediv is implemented here (as opposed to truediv and floordiv). Therefore, using the '/' operator will return a floating-point approximation, not the truncated down result of '//'
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real / other)
            self.child.append((1.0 / other, f))
        else:
            f = ReverseMode(self.real / other.real)
            other.child.append((-self.real / (other.real)**2, f))
            self.child.append((1.0 / other.real, f))
        return f


    def __rtruediv__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for division operator in reverse case (b / a)
        
        Inputs
        ------------------------------------
        self: object by which the first object is divided; a in b/a
              ReverseMode object, int, or float
        other: object being divided; b in b/a
               ReverseMode object
        
        Outputs
        ------------------------------------
        x = b/a
        a ReverseMode object with the value of the other/self operation
        
        Examples
        ------------------------------------
        int/ReverseMode:
        >>> x = 6/ReverseMode(3)
        >>> print(x.real)
        2.0
        
        ReverseMode/ReverseMode:
        >>> x = ReverseMode(6)/ReverseMode(2)
        >>> print(x.real)
        3.0
        
        Notes
        ------------------------------------
        Only rtruediv is implemented here (as opposed to rtruediv and rfloordiv). Therefore, using the '/' operator will return a floating-point approximation, not the truncated down result of '//'
        '''
        
        f = ReverseMode(other / self.real)
        self.child.append((other * (-self.real ** (-2)), f))
        return f

    def __neg__(self):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for negative operator (-a)
        
        Inputs
        ------------------------------------
        self: object raised to an exponent; a in a**b
              ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = -a
        a ReverseMode object with the value of the -self operation
        
        Examples
        ------------------------------------
        -ReverseMode:
        >>> x = -ReverseMode(5)
        >>> print(x.real)
        -5
        '''
        
        if not isinstance(self, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        else:
            f = ReverseMode(-self.real)
            self.child.append((-1, f))
        return f

    def __pow__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for power operator (a**b)
        
        Inputs
        ------------------------------------
        self: object raised to an exponent; a in a**b
              ReverseMode object
        other: object that is the exponent; b in a**b
               ReverseMode object, int, or float
        
        Outputs
        ------------------------------------
        x = a**b
        a ReverseMode object with the value of the self**other operation
        
        Examples
        ------------------------------------
        ReverseMode**int:
        >>> x = ReverseMode(5)**2
        >>> print(x.real)
        25
        
        ReverseMode**ReverseMode:
        >>> x = x = ReverseMode(5)**ReverseMode(2)
        >>> print(x.real)
        25
        '''
        
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real ** other)
            self.child.append((other * (self.real ** (other - 1.0)), f))
        else:
            f = ReverseMode(self.real ** other.real)
            other.child.append((self.real ** other.real * np.log(self.real), f))
            self.child.append((other.real * self.real ** (other.real - 1.0), f))
        return f

    def __rpow__(self, other):
        '''
        Explanation
        ------------------------------------
        Overloaded dunder method for power operator in reverse case (b**a)
        
        Inputs
        ------------------------------------
        self: object that is the exponent; a in b**a
              ReverseMode object, int, or float
        other: object that is raised to an exponent; b in b**a
               ReverseMode object
        
        Outputs
        ------------------------------------
        x = b**a
        a ReverseMode object with the value of the other**self operation
        
        Examples
        ------------------------------------
        int**ReverseMode:
        >>> x = 2**ReverseMode(5)
        >>> print(x.real)
        32
        
        ReverseMode**ReverseMode:
        >>> x = x = ReverseMode(5)**ReverseMode(2)
        >>> print(x.real)
        25
        '''
        
        f = ReverseMode(other ** self.real)
        self.child.append(((other ** self.real) * np.log(other) , f))
        return f
