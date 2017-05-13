"""Common functions used by GraphESN and learning procedures.

State Mapping Functions:
    mean_state_mapping
    supersource_state_mapping
    sum_state_mapping

Activation Functions:
    hyperbolic_tangent  --  Hyperbolic tangent with some parameter to modify
                            the slope or the output value.
    tanh                --  Standard hyperbolic tangent.
    linear              --  Identity function.

Generic Functions:
    step_function
    double_step_function
    sign

"""

import scipy
import math
from math import copysign

#---------------------------------- State Mapping Functions

def mean_state_mapping(xg):
    """Apply the mean state mapping function on given internal state 
    representation.
    
    How it works: return the average vector.

    Arguments:
    xg  --  Internal representation of graph g. (scipy.array (|V(g)|,Nr))

    Return:
    Obtained reservoir activations (scipy.array (Nr,)).

    """
    return scipy.sum(xg, axis=0) / xg.shape[0]

def supersource_state_mapping(xg, index=-1):
    """Apply the supersource state mapping function.

    How it works: return the encoding of the first vertex.

    Arguments:
    xg      --  Internal representation of graph g. (scipy.array (|V(g)|,Nr))
    index   --  Specify the index to select [def: 0]

    Return:
    Obtained reservoir activations (scipy.array (Nr,)).

    """
    return xg[index]


def sum_state_mapping(xg):
    """Return the sum of all activations for each vertex.

    Differs from the mean state mapping by taking in account the size of the 
    graph.

    Arguments:
    xg  --  Internal representation of graph g. (scipy.array (|V(g)|,Nr))

    Return:
    Obtained reservoir activations (scipy.array (Nr,)).

    """
    return scipy.sum(xg, axis=0)

#---------------------------------- Activation Functions

class ActivationFunction(object):
    """Metaclass used to define activation functions.
    Each activation function is callable and provides methods to call its
    inverse (inv) and its derivative (der).

    """

    def __init__(self, fun, inv, der, descr="ActivationFunction"):
        """Init new activation function.
        
        Arguments:
        fun --  Function to apply when called.
        inv --  Inverse function.
        der --  Function derivative.         
    
        """
        self.fun = fun
        self.inv = inv
        self.der = der
        self.descr = descr

    def __call__(self, v):  
        res = self.fun(v)
        assert (not scipy.isnan(res).any()) and (not scipy.isinf(res).any()),\
            "Activation function returned NaN or Infty"
        return res

    def inv(self, v):       
        res = self.inv(v)
        assert (not scipy.isnan(res).any()) and (not scipy.isinf(res).any()),\
            "Activation function's inverse returned NaN or Infty"
        return res

    def der(self, v):       
        res = self.der(v)
        assert (not scipy.isnan(res).any()) and (not scipy.isinf(res).any()),\
            "Activation function's derivative returned NaN or Infty"
        return res

    def __repr__(self):
        return self.descr

def hyperbolic_tangent(a, b, prime_offset=0.0, threshold=float('inf')):
    """Return an element-wise hyperbolic tangent with range 'a' and slope 'b'.

    i.e. f(x) = a * tanh(b * x)

    Arguments:
    a               -   Output range: [-a, +a]. (real)
    b               -   Slope parameter. (real)
    prime_offset    -   Add a constant value to the derivative, avoiding values
                        to stuck on zero for large input. (real)
    threshold       -   When the output's magnitude is greater than the 
                        threshold then return 'a' (or '-a'). (real)

    Return:
    An ActivationFunction object.

    """
    thr_fun = lambda X: (X < -threshold) * -a + (X > threshold) * a + ((X < -threshold) + (X > threshold) == 0) * X
    fun = lambda X: thr_fun(a * scipy.tanh(X * b))
    # der = lambda X: scipy.ones(X.shape) - scipy.tanh(X)**2
    ab = a * b
    der = lambda X: ab * (scipy.ones(X.shape) - scipy.tanh(X * b)**2) + scipy.ones(X.shape) * prime_offset
    inv = lambda X: scipy.arctanh(X / a) / b
    descr = "hyperbolic_tangent(%f, %f, %f, %f)" % (a, b, prime_offset, threshold)
    return ActivationFunction(fun, inv, der, descr)


# Standard Hyperbolic Tangent function
# D[tanh(x)] = 1 / cosh^2(x) = 1 - tanh^2(x)
tanh = ActivationFunction(
        scipy.tanh, 
        scipy.arctanh, 
        lambda X: scipy.ones(X.shape) - scipy.tanh(X)**2, 
        "tanh"
    )
tanh.__doc__ = """Element-wise hyperbolic tangent (ActivationFunction)."""

# Linear activation function
linear = ActivationFunction(
        lambda X: X,
        lambda X: X,
        lambda X: scipy.ones(X.shape),
        "linear"
    )

#---------------------------------- Generic Functions (e.g Output Functions)

def step_function(value=1.0):
    """Return an elementwise step function.

    Arguments:
    value   --  Return value for positive inputs (real)
    
    Return:
    An elementwise step function.

    """
    f = lambda X: (X < 0.0) * -value + (X >= 0.0) * value
    return f

def double_step_function(value=1.0, epsilon=0.5): 
    """Return an elementwise (double) step function. 
    
    Arguments:
    value   --  Return value for positive inputs (real)
    epsilon --  The threshold (a positive real value).

    Return:
    An elementwise function such that f(v_i) is
        +value  :   if v_i > +epsilon
         0      :   if -epsilon <= v_i <= +epsilon
        -value  :   if v_i < -epsilon 

    """
    f = lambda x: -value if x < -epsilon else +value if x > epsilon else 0.0
    return scipy.vectorize(f)

def right_shoulder(threshold=1.0):
    """An element-wise right-shoulder function:

    f(x) = threshold if x > threshold
    f(x) = x if -threshold <= x <= threshold
    f(x) = -threshold if x < -threshold

    Arguments:
    threshold   --  Threshold value (real).
    
    Return:
    The element-wise right-shoulder function.

    """
    f = lambda x: copysign(threshold, x) if abs(x) > threshold else x
    return scipy.vectorize(f)

def max_value_classifier((pos, neg)=(1.0,-1.0)):
    """Return a function that transform an input vector into a class, expressed
    by 1-of-k enccoding, by watching the maximum value.

    E.g. With bipolar encoding {-1.0, +1.0}

        f([0.2, 0.7, -0.1, 0.5]) = [-1.0, +1.0, -1.0, -1.0]

    Arguments:
    (pos, neg)  -- positive and negative value of the encoding (real).

    Return:
    A function callable on scipy.arrays: f :: (N,) -> (N,).

    """
    def f(v):
        r = [neg] * len(v)
        idx = v.tolist().index(max(v))        
        r[idx] = pos
        return scipy.array(r)
    return f

def sign(v) :
    """Elementwise sign function.

    Arguments:
    v   --  Input (scipy.array)
    
    Return:
    A scipy.array 'r' such that r_i = cmp(v_i, 0).

    """
    return scipy.vectorize(lambda x : cmp(x, 0))(v)    
