"""Provide functions for models trainining."""

from itertools import product

import pdb
import scipy
import scipy.linalg as linalg
from sys import float_info

def lms(network, dataset, label, Target, eta=0.1, momentum=0.0, 
        threshold=10**-5, patience=10, maxit=5000, decay=0.0):
    """Perform the iterative Least Mean Squares algorithm.

    The algorithm performs a gradient descent to minimize the sum of the 
    squared errors.    

    Arguments:
    network     --  Model to be trained.
    dataset     --  Training-set (list of g_esn.graph.Graph)
    label       --  Name of the input label (string).
    Target      --  Target matrix (scipy.array (len(dataset), Ny))
    eta         --  Learning rate (real)
    momentum    --  Momentum (real)
    threshold   --  Minimum variation to consider the algorithm not converged 
                    yet (real).
    patience    --  Patience factor (int)
    maxit       --  Maximum iterations allowed (int)
    decay       --  Weight decay factor.

    Return:
    sq_error    --  Last squared error (real).
    epochs      --  Number of iteration performed (int).
    W           --  Obtained weights matrix (scipy.array (Ny, Nf))

    """
    # collect the input
    Input = scipy.array([network.get_activations(g, label) for g in dataset])
    Input = scipy.concatenate((Input, scipy.ones((Input.shape[0],1))), 1) # bias
    # some check
    Ns = len(dataset) # number of samples
    Ny = Target.shape[1] # number of outputs
    Nf = Input.shape[1] # number of features (including the bias)
    assert Input.shape == (Ns, Nf), "Invalid shape. I:%s" % str(Input.shape)
    assert Target.shape == (Ns, Ny), "Invalid shape. E:%s" % str(Target.shape)
    assert (not scipy.isnan(Input).any()) and (not scipy.isinf(Input).any()),\
                "Some input is NaN or Infty"
    assert (not scipy.isnan(Target).any()) and (not scipy.isinf(Target).any()),\
                "Some target is NaN or Infty"
    # perform iterative fitting
    best_error = float_info.max
    quit_epoch = patience
    epoch = 0
    last_delta = scipy.zeros((1,Nf)) # last computed weight change
    W = scipy.random.uniform(-1.0, 1.0, network.W_out.shape) # (Ny,Nf)
    Net = scipy.dot(Input, W.T) # (Ns,Ny)
    Out = network.out_act(Net) # (Ns,Ny)
    while epoch < quit_epoch and epoch < maxit:
        # slope: \partial E / \partial w        
        primeF = network.out_act.der(Net) # (Ns,Ny)
        assert (not scipy.isnan(primeF).any()) and (not scipy.isinf(primeF).any()),\
                    "Some derivative is NaN or Infty"
        EP = (Target - Out) * primeF # (Ns,Ny)
        slope = [] 
        for o in xrange(Ny):
            sl_o = []
            for j in xrange(Nf):
                sl_o.append(-scipy.sum(EP[:,o] * Input[:,j]))
            slope.append(sl_o)
        # weights update        
        gradient = - scipy.array(slope) # (Ny,Nf)
        delta = (gradient * eta) + (last_delta * momentum)
        W = (W + delta) - (W * decay)
        last_delta = delta
        # current error
        Net = scipy.dot(Input, W.T) # (Ns,Ny)
        Out = network.out_act(Net) # (Ns,Ny)
        errors = Target - Out
        sq_error = scipy.sum(0.5 * (errors**2))
        # update stop condition
        if (best_error - sq_error) / sq_error > threshold:
            best_error = sq_error
            quit_epoch = epoch + patience
        epoch += 1
    # replace current readout
    network.W_out = W
    return sq_error, epoch, W

def least_squares(network, dataset, label, Target):
    """Train given model by minimizing the sum of the squared residual errors.
    Find global minima by using the direct solution (uses Moore-Penrose 
    pseudoinverse.

    Arguments:
    network --  GraphESN to train.
    dataset --  Training-set. (list of g_esn.graph.Graph)
    label   --  Attribute name of the input label. (string)
    Target  --  Target values matrix. (scipy.array (Ns,Ny))

    Return:
    Assuming Ns the number of input samples/pattern, Nf the number of features
    (bias included) for each input and Ny the output dimension:
    Input   --  Input matrix (Ns, Nf)
    Target  --  Inverted target values (Ns, Ny)
    W       --  Weights matrix (Nf, Ny). 
                Warning: W.T is the actual weights matrix used by a GraphESN.

    ----------
    How it works:
    Let 'Ns' be the number of samples in the dataset, 'Nf' the number of
    features excracted (e.g. by the reservoir) for each sample, including a bias 
    term, and 'Ny' the number of outputs.
    Then output weights are calculated as
        
        W = (pseudoinverse(Input) * network.out_act.inv(Target)).T

    where:
        Input   : contains the all inputs (Ns, Nf)
        Target  : contains target values (Ns, Ny)
        W.T     : weights matrix (Ny, Nf)
        
    Inverse of the output activation function is applied to target values, in 
    order to properly fit the output. Thus the network is espected to have an 
    invertible output activation function and target values are expected to
    properly fall into the function's range.
    (see also g_esn.functions.ActivationFunction)    

    """
    # collect the input
    I = scipy.array([network.get_activations(g, label) for g in dataset])
    I = scipy.concatenate((I, scipy.ones((I.shape[0],1))), 1) # bias
    # modify the target (apply inverse of the output function)
    T = network.out_act.inv(Target)
    # some check
    Ns = len(dataset) # number of samples
    Ny = T.shape[1] # number of outputs
    Nf = I.shape[1] # number of features (including the bias)
    assert I.shape == (Ns, Nf), "Invalid shape. I:%s" % str(I.shape)
    assert T.shape == (Ns, Ny), "Invalid shape. E:%s" % str(T.shape)
    assert (not scipy.isnan(I).any()) and (not scipy.isinf(I).any()),\
                "Some input is NaN or Infty"
    assert (not scipy.isnan(T).any()) and (not scipy.isinf(T).any()),\
                "Some target is NaN or Infty"
    # perform fitting: Input * W = Target, with Input:(Ns,Nf), Target:(Ns,Ny)
    W = scipy.dot(scipy.linalg.pinv(I), T)
    assert W.T.shape == network.W_out.shape,\
        "Wrong shape in learned matrix: %s" % str(W.shape)
    network.W_out = W.T # replace current weight matrix
    return (I, T, W)


def ridge_regression(network, dataset, label, Target, alpha):
    """Train given model using Ridge-Regression algorithm to solve the LMS 
    problem.   

    Arguments:
    network --  GraphESN to train.
    dataset --  Training-set. (list of g_esn.graph.Graph)
    label   --  Attribute name of the input label. (string)
    Target  --  Target values matrix. (scipy.array (Ns, Ny))
    alpha   --  Ridge-regression coefficient (see below). (real)

    Return:
    Assuming Ns the number of input samples/pattern, Nf the number of features
    (bias included) for each input and Ny the output dimension:
    Input   --  Input matrix (Ns, Nf).
    Target  --  Inverted target values (Ns, Ny)
    W       --  Weights matrix (Nf, Ny). 
                Warning: W.T is the actual weights matrix used by a GraphESN.

    ----------
    How it works (see [1]):
    Let 'Ns' be the number of samples in the dataset, 'Nf' the number of
    features excracted (e.g. by the reservoir) for each sample, including a bias 
    term, and 'Ny' the number of outputs.
    Then output weights are calculated as

        W = ((Input^T Input + \alpha I)^-1 Input^T out_act.inv(Target)).T

    where:
        Input   : contains all inputs (Ns, Nf)
        Target  : contains target values (Ns, Ny)
        W.T     : weights matrix (Ny, Nf)
    
    Inverse of the output activation function is applied to target values, in 
    order to properly fit the output. Thus the network is espected to have an 
    invertible output activation function and target values are expected to
    properly fall into the function's range.
    (see also g_esn.functions.ActivationFunction)

    ----------
    [1] T. Hastie, R. Tibshirani, J. H. Friedman. The Elements of Statistical
        Learning. Springer-Verlag, 2001.

    """    
    # collect the input
    I = scipy.array([network.get_activations(g, label) for g in dataset])
    I = scipy.concatenate((I, scipy.ones((I.shape[0],1))), 1) # bias
    # modify the target (apply inverse of the output function)
    T = network.out_act.inv(Target)
    # some check
    Ns = len(dataset) # number of samples
    Ny = T.shape[1] # number of outputs
    Nf = I.shape[1] # number of features (including the bias)
    assert I.shape == (Ns, Nf), "Invalid shape. I:%s" % str(I.shape)
    assert T.shape == (Ns, Ny), "Invalid shape. E:%s" % str(T.shape)
    assert (not scipy.isnan(I).any()) and (not scipy.isinf(I).any()),\
                "Some input is NaN or Infty"
    assert (not scipy.isnan(T).any()) and (not scipy.isinf(T).any()),\
                "Some target is NaN or Infty"
    # perform fitting: Input * W = Target. Input:(Ns,Nf), Target:(Ns,Ny)    
    W = scipy.dot(
        linalg.inv(scipy.dot(I.T, I) + alpha * scipy.eye(I.shape[1]), True), 
        scipy.dot(I.T, T)
    )
    assert W.T.shape == network.W_out.shape,\
                "Wrong shape in learned matrix: %s" % str(W.shape)
    network.W_out = W.T # replace current readout
    return (I, Target, W.T) 

def max_S(network, dataset, label, Errors, eta=0.1, momentum=0.0, 
        threshold=10**-5, patience=10, maxit=5000, decay=0.0):
    """Train the readout weights matrix in order to maximize the correleation
    with the residual error of the base-network, S.

    S = \sum_{o \in out} | \sum_{g \in D}(z(g) - \bar{z})(e_o(g) - \bar{e}_o) |

    Maximization is acheived by performing a gradient ascent as described 
    in [1]. In order to evaluate the gradient, if the activation function is not
    None, then is to be invertible so is expected to have a 'der' method.

    Arguments:
    network     --  GraphESN to train.
    dataset     --  Training-set. (list of g_esn.graph.Graph)
    label       --  Attribute name of the input label. (string)
    Errors      --  Residual errors (scipy.array (Ns, Ny))
    eta         --  Gradient ascent step size. (real) 
    momentum    --  Momentum scaling in weights update. (real)
    threshold   --  Stop learning when the percentage increase of S (gain) 
                    is lower than given threshold. (real in [0,1]) 
    patience    --  A 'patience' parameter. Each time the best S value overcome
                    the threshold, the maximization process is allowed to
                    perform a number of extra epochs equals to this value.
    maxit       --  Maximum number of iterations allowed to maximize S. (int)
    decay       --  Weight decay factor (real).

    Return:
    S   --  Correlation value: S. (real number).
    z   --  Outputs, scipy.array (Ns, 1).
    it  --  Number of performed iterations (int).

    ----------
    [1] Scott E. Fahlman and Christian Lebiere. The cascade-correlation learning
        architecture. In Advances in Neural Information Processing Systems 2, 
        pages 524-532. Morgan Kaufmann, 1990.

    """
    # candidate's input
    I = scipy.array( [network.get_activations(g, label) for g in dataset] )
    I = scipy.concatenate((I, scipy.ones((I.shape[0], 1))), 1) # (Ns, Nf)
    # some useful number
    Ny = Errors.shape[1] # number of network outputs
    Ns = len(dataset) # number of patterns
    Nf = I.shape[1] # number of inputs (e.g. reservoir activations + bias)
    # current weights matrix    
    W = network.W_out[:] # candidate's weights matrix (1, Nf)
    # some check    
    assert I.shape == (Ns, Nf),      "Invalid shape. A:%s" % str(I.shape)
    assert Errors.shape == (Ns, Ny), "Invalid shape. Errors:%s" % str(Errors.shape)
    assert W.shape == (Ny, Nf),      "Invalid shape. W:%s" % str(W.shape)
    # learning
    avgE = scipy.sum(Errors, 0) / Ns # average error (Ny,)
    EE = Errors - avgE # calculate once. (Ns, Ny)    
    # iterative gradient ascent
    best_S = 0.0 # best measured S
    quit_epoch = patience
    epoch = 0 # epoch counter
    last_delta = scipy.zeros((1,Nf)) # last computed weight change
    while epoch < quit_epoch and epoch < maxit:
        # unit's output
        Net = scipy.dot(I, W.T) # (Ns, 1)
        primeF = network.out_act.der(Net) # (Ns, 1)  
        Z = network.out_act(Net) # candidate's output (Ns, 1)
        avgZ = scipy.sum(Z, 0) / Ns # average output: (1,)
        ZZ = Z - avgZ # calculate once. (Ns, 1)
        # correlation (sigma)
        corr = scipy.dot(ZZ.T, EE) # (1, Ny)
        Sigma = scipy.sign(corr) # (1, Ny)
        # new S
        S = scipy.sum(scipy.absolute(corr))
        # slope: \partial E / \partial w
        SEEP = (Sigma * EE * primeF) # evaluate only once (Ny, Ns)
        gradient = scipy.dot(scipy.sum(SEEP, 1), I)
        # update weights
        delta = gradient * eta + last_delta * momentum
        W = (W + delta) - (W * decay)
        last_delta = delta
        # stop condition
        if (S - best_S) / best_S > threshold:
            best_S = S
            quit_epoch = epoch + patience
        epoch += 1
    # replace current readout
    network.W_out = W
    Z = network.out_act(scipy.dot(I, W.T))
    return (S, Z, epoch)

