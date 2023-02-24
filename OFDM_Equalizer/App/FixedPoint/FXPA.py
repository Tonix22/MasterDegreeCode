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
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            # Handle slicing
            return FXPA(self.data[index])
        else:
            # Handle mask index
            mask = np.asarray(index, dtype=bool)
            return FXPA(self.data[mask])
    
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
            get = HANDLED_FUNCTIONS[ufunc](*scalars, **kwargs)
            if isinstance(get, numpy.bool_):
                return get
            else:
                return self.__class__(get)
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
        return np.sum(arr._N).like(MULTIPLIERS)

    @implements(np.add)
    def add(arr1, arr2):
        sum = (arr1 + arr2).like(MULTIPLIERS)
        return sum

    @implements(np.multiply)
    def multiply(arr1, arr2):
        mul = (arr1*arr2).like(MULTIPLIERS)
        return mul
    
    @implements(np.greater_equal)
    def greater_equal(arr1, arr2):
        geq = arr1.astype(type(arr2)) >= arr2
        return geq

    @implements(np.less_equal)
    def less_equal(arr1, arr2):
        geq = arr1.astype(type(arr2)) <= arr2
        return geq
    
    @implements(np.maximum)
    def maximum(arr1, arr2):
        fmax = max(arr1.astype(float), arr2.astype(float))
        return fmax
    
    @implements(np.minimum)
    def minimum(arr1, arr2):
        fmax = min(arr1.astype(float), arr2.astype(float))
        return fmax
    
    

# Define a vectorized constructor for FXPA objects
fxpa_build = np.vectorize(FXPA)

class FXP_Linear:
    def __init__(self, weight, bias):
        self.weight = fxpa_build(weight)
        self.bias   = fxpa_build(bias)

    def __call__(self, x):
        return np.dot(x, self.weight.T) + self.bias

class HardTanh:
    def __init__(self, min_val=-1, max_val=1):
        self.min_val = FXPA(min_val)   
        self.max_val = FXPA(max_val)
    
    def cut(self,a):
        for n in range(0,len(a)):
            a[n] = FXPA(max(a[n]._N.astype(float), self.min_val._N.astype(float)))
            a[n] = FXPA(min(a[n]._N.astype(float), self.max_val._N.astype(float)))
        return a
        
    def __call__(self, x):
        return self.cut(x)
    
class FXP_Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

"""
weight = np.array([[1.11, -1.77], [2.333, -3.14]])
bias   = np.array([-3.33, 3.1614])
linear1 = FXP_Linear(weight,bias)
linear2 = FXP_Linear(weight,bias)
activation = HardTanh()

x       = fxpa_build(np.array([0.2,.04]))
out     = linear1(x)
out     = activation(out)

print(out)
"""