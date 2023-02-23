import numpy as np    
from numbers import Number        
import numpy.lib.mixins
from fxpmath import Fxp
import numpy as np

MULTIPLIERS = Fxp(1, signed=True, n_word=16, n_frac=11)       

HANDLED_FUNCTIONS = {}
class FXPA(numpy.lib.mixins.NDArrayOperatorsMixin):

    def __init__(self, N):
        if isinstance(N, Fxp):
            self._N = N.like(MULTIPLIERS)
        else:
            self._N = Fxp(N, signed=True, n_word=16, n_frac=11).like(MULTIPLIERS)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(N={self._N})"

    def __array__(self, dtype=None):
        return np.array(self._N)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    
        if method == '__call__':

            N = None

            scalars = []

            for input in inputs:

                # In this case we accept only scalar numbers or FXPAs.

                if isinstance(input, Number):

                    scalars.append(input)

                elif isinstance(input, self.__class__):

                    scalars.append(input._N)

                    if N is not None:

                        if N != self._N:

                            raise TypeError("inconsistent sizes")

                    else:

                        N = self._N

                else:

                    return NotImplemented

            #return self.__class__(ufunc(*scalars, **kwargs))
            return self.__class__(HANDLED_FUNCTIONS[ufunc](*scalars, **kwargs))

        else:

            return NotImplemented

def implements(np_function):
   "Register an __array_function__ implementation for FXPA objects."
   def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
   return decorator

@implements(np.sum)
def sum(arr):
    "Implementation of np.sum for FXPA objects"
    return np.sum(arr._N)

@implements(np.add)
def add(arr1, arr2):
    sum = (arr1 + arr2).like(MULTIPLIERS)
    return sum

@implements(np.multiply)
def multiply(arr1, arr2):
    mul = (arr1*arr2).like(MULTIPLIERS)
    return mul

arr = [] 
arr.append(FXPA(1.33))
arr.append(FXPA(0.27))
a = np.array([[FXPA(1.11), FXPA(1.77)], [FXPA(0.333), FXPA(3.14)]])
b = np.array([FXPA(0.01), FXPA(1.11)])
c = a@b
print(c)
#print(np.sum(np.asarray(arr)))