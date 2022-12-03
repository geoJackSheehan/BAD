import numpy as np

class ReverseMode():
    _supported_scalars = (int, float)

    def __init__(self, real):
        self.real = real
        self.child = []
        self.grad = None

    def gradient(self):
        if self.grad is None:
            for dvj_dvi, df_dvj in self.child:
                self.grad += dvj_dvi * df_dvj.gradient()
        return self.grad

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
        else:
            f = ReverseMode(self.real * other.real)
            other.child.append((self.real, f))
            self.child.append((other.real, f))
        return f

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
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
        f = ReverseMode(other / self.real)
        self.child.append((other * (-self.real ** (-2)), f))
        return f

    def __neg__(self):
        if not isinstance(self, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        else:
            f = ReverseMode(-self.real)
            self.child.append((-1, f))
        return f

    def __pow__(self, other):
        if not isinstance(other, (*self._supported_scalars, ReverseMode)):
            raise TypeError('Type not supported: must be int or float')
        if isinstance(other, self._supported_scalars):
            f = ReverseMode(self.real ** other)
            self.child.append((other * (self.real ** (other - 1.0)), f))
        else:
            f = Rnode(self.real ** other.real)
            other.child.append((self.real ** other.real * np.log(self.real), f))
            self.child.append((other.real * self.real ** (other.real - 1.0), f))
        return f

    def __rpow__(self, other):
        f = ReverseMode(other ** self.real)
        self.child.append(((other ** self.real) * np.log(other) , f))
        return f
