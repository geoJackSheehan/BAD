'''
This module defines dunder methods to overload Python built-in operators in the Dual class.
'''

class DualNumber:
    
    _supported_scalars = (int,float)
    
    def __init__(self,real,dual=1.0):
            self.real = real
            self.dual = dual
    
    def __add__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other,self._supported_scalars):
            return DualNumber(other+self.real,self.dual)
        else:
            return DualNumber(self.real+other.real,self.dual+other.dual)
        
    def __radd__(self,other):
        return self.__add__(other)
    
    def __sub__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other,self._supported_scalars):
            return DualNumber(self.real-other,self.dual)
        else:
            return DualNumber(self.real-other.real,self.dual-other.dual)
    
    def __rsub__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float")
        if isinstance(other,self._supported_scalars):
            return DualNumber(other-self.real,-self.dual)
        else:
            return DualNumber(-self.real+other.real,-self.dual+other.dual)
    def __mul__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float") 
        if isinstance(other,self._supported_scalars):
            return DualNumber(self.real*other.real,self.dual*other.real)
        else:
            return DualNumber(self.real*other.real,self.real*other.dual+other.real*self.dual)
        
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __truediv__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float") 
        if isinstance(other,self._supported_scalars):
            return DualNumber(self.real/other.real,self.dual/other.real)
        else:
            return DualNumber(self.real/other.real,(other.real*self.dual - self.real*other.dual)/(other.real*other.real))
        
    def __rtruediv__(self,other):
        if not isinstance(other,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float") 
        if isinstance(other,self._supported_scalars):
            return DualNumber(other.real/self.real,(-other.real*self.dual)/(self.real*self.real))
        else:
            return DualNumber(other.real/self.real,-(other.real*self.dual - self.real*other.dual)/(other.dual*other.dual))
        
    def __neg__(self):
        if not isinstance(self,(*self._supported_scalars, DualNumber)):
            raise TypeError("Type not supported: must be int or float") 
        if isinstance(self,self._supported_scalars):
            return DualNumber(self.real*(-1),self.dual*(-1))
        else:
            return DualNumber(self.real*(-1),self.dual*(-1))
        
    def __pow__(self,other):
        raise NotImplementedError
    
    def __rpow__(self,other):
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
    
    