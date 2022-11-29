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
