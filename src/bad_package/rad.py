import numpy as np

class ReverseMode():
    _supported_scalars = (int, float)

    def __init__(self, real):
        self.real = real
        self.child = []
        self.grad = None

    def __add__(self, other):
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
        return self.__add__(other)

    def __sub__(self, other):
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
        return self.__sub__(other)

    def __mul__(self, other):
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real * other)
            self.child.append((other, f))
            other.child.append((self.real, f))
        else:
            f = ReverseMode(self.real * other.real)
            self.child.append((other.real, f))
        return f
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            new_other = ReverseMode(1 / other.real)
            other.child.append((-other.real ** -2, new_other))
            return self * new_other
        else:
            f = ReverseMode(self.real / other)
            grad = 1 / other
            self.child.append((grad, f))
    
    def __rtruediv__(self, other):
        f = ReverseMode(other / self.real)
        grad = other * (-self.real ** (-2))
        self.child.append((grad, f))
        return f

    def __neg__(self):
        f = ReverseMode(-self.real)
        self.child.append((-1, f))
        return f

    def __pow__(self, exponent):
        # TODO
        pass

    def __pow__(self, other):
        # TODO
        pass



