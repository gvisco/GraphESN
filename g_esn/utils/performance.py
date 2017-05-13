"""Module implements functions to manage trained models in order to get
informations about their performances (e.g. misclassification error, confusion
matrices, residual errors) or to exctract useful data based on their execution
(collect errors to be used as subnetwork's target or perform a PCA on 
reservoir's activations).

"""

import scipy
from scipy.linalg import norm, svd
from g_esn.graph_esn import *


def evaluation_details(gesn, dataset, label, target):
    """Let the model evaluate given dataset and return a list of pairs:

        {(y(g), y_target(g)) | g in dataset}

    Arguments:
    gesn    --  Model to evaluate (GraphESN).
    dataset --  Dataset (list of Graph objects).
    label   --  Name of the input label, as graph attribute (string).
    target  --  Attribute name of the target label
                OR
                Target values matrix (scipy.array (Ns,Ny))

    Return:
    A list of pairs {(y(g), y_target(g)) | g in dataset}, where y, y_target
    are scipy.array shaped (Ny,).

    """
    if isinstance(target, str):
        return [(gesn.evaluate(g, label), g.attr[target]) for g in dataset]
    elif isinstance(target, scipy.ndarray): # explicit target given
        return [(gesn.evaluate(g, label), target[i]) for i, g in enumerate(dataset)]
    else:
        raise Exception("Invalid argument 'target': %s" % str(target))
    
def _as_tuples(details):
    """Convert the details in a list of tuples instead of scipy.arrays.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).

    Return:
    The list of details {(y(g), y_target(g)) | g in dataset} with y, y_target
    as tuples of length Ny.
    
    """
    f = lambda (o, t): (tuple(o), tuple(t))
    return map(f, details)

def pca(data, dim=2):
    """Perform the Principal Component Analysis (PCA) through SVD ad return the
    samples projected over the principal components.
    When 'dim' is 2 (or 1, or 3) resulting data can be easily plotted.

    Arguments:
    data    --  scipy.array (N,M) with N samples and M features.
    dim     --  Number of principal components to be taken (int).

    Return:
    A scipy.array shaped (N,dim) containing all the N samples projected over 
    the first 'dim' principal components.

    --------------- 

    Reference:
    A Tutorial on Principal Component Analysis, by Jonathon Shlens.
    
    Test:
    Dataset: 
        http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    Results: 
        http://scikit-learn.sourceforge.net/0.5/auto_examples/plot_pca.html

    """
    from math import sqrt
    N, M = data.shape # N: trials; M: dimensions
    if dim > M: 
        raise Exception("Wrong param: 'dim'. Data doesn't have enough dimensions.")
    mean = scipy.mean(data, axis=0) # mean for each column
    data = data - mean # center the data
    Y = data / sqrt(N-1) # also: Y = data / data.std()
    U, S, PC = svd(Y, full_matrices=False) # SVD 
    points = U[:,:dim] * S[:dim] # project data
    return points

def reservoir_pca(gesn, dataset, label, target, dim=2):
    """Perform the Principal Component Analysis on the reservoir's activations.

    For each graph 'g' in the dataset, X(g) is evaluated (method 
    GraphESN.get_activations) then PCA is performed and data projected over the 
    principal components is returned.
    When 'dim' is 2 (or 1 or 3) returned data can be easily plotted in order to 
    have a visual representation of how the reservoir works with different
    inputs.
    
    Arguments:
    gesn    --  Model to evaluate (GraphESN).
    dataset --  Dataset (list of Graph objects).
    label   --  Name of the input label, as graph attribute (string).
    target  --  Attribute name of the target label
                OR
                Target values matrix (scipy.array (Ns,Ny))
    dim     --  Number of dimensions of the projected data (i.e. number of
                principal components to take in account). (int)

    Return:
    A list of pairs (projection, target), one for each pattern in the dataset,
    where 'projection' is a 'dim' sized tuple resulting from the projection of
    the data over the principal components and 'target' is the target value
    associated with given input graph.

    """
    # perform PCA
    activations = [gesn.map_fun(gesn.encode(g, label)) for g in dataset]
    activations = scipy.array(activations) # array (|D|, Nr)
    points = pca(activations, dim)
    # pack with the target and return
    if isinstance(target, str):
        return [(tuple(points[idx]), g.attr[target]) for idx, g in enumerate(dataset)]
    elif isinstance(target, scipy.ndarray): # explicit target given
        return [(tuple(points[idx]), target[idx]) for idx, g in enumerate(dataset)]
    else:
        raise Exception("Invalid argument 'target': %s" % str(target))

def misclassification(details):
    """Evaluate the classification performance of a GraphESN over given dataset.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    
    Return:
    m_rate  --  Misclassification rate (real number in [0,1])
    matrix  --  The confusion matrix as bidimensional list (actual values on 
                the rows, predicted values on the columns)    
    classes --  A list containing values for any class (list).

    """
    details = _as_tuples(details) # convert to deal with comparisons
    misclass = 0.0
    cl = sorted(set([d[1] for d in details])) # collect classes
    num_cl = len(cl)
    len_data = len(details)
    matrix = [[0.0] * num_cl for _ in range(num_cl)] # fill matrix with zeros
    for (out, tar) in details:
        actual = cl.index(tar)
        pred = cl.index(out)
        matrix[actual][pred] += 1.0 / len_data
        if not actual == pred:
            misclass += 1.0
    m_rate = misclass / len_data # normalize: misclassification-rate
    return m_rate, matrix, cl

def confusion_matrix(details):
    """Get the confusion matrix describing the classification 
    performance of a GraphESN over given dataset.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    
    Return:
    matrix  --  The confusion matrix as bidimensional list (actual values on 
                the rows, predicted values on the columns)    
    classes --  A list containing values for any class (list).

    """
    details = _as_tuples(details) # convert to deal with comparisons
    cl = sorted(set([d[1] for d in details])) # collect classes
    num_cl = len(cl)
    len_data = len(details)
    matrix = [[0.0] * num_cl for _ in range(num_cl)] # init matrix to zero
    for (out, tar) in details:
        actual = cl.index(tar)
        pred = cl.index(out)
        matrix[actual][pred] += 1.0 / len_data
    return matrix, cl

def misclassification_rate(details):
    """Get the misclassification rate of a GraphESN over given dataset.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    
    Return:
    Misclassification-rate (real number in [0,1]).

    """
    details = _as_tuples(details) # convert to deal with comparisons
    misclass = 0.0
    cl = sorted(set([d[1] for d in details])) # collect classes
    for (out, tar) in details:
        actual = cl.index(tar)
        pred = cl.index(out)
        if not actual == pred:
            misclass += 1.0
    m_rate = misclass / len(details) # normalize: misclassification-rate
    return m_rate
    
def mean_absolute_error(details):
    """Get the average absolute error made by the network.

        err = 1/|D| \sum_{g in D} ||y_target(g) - y(g)||

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).

    Retun:
    Computed mean absolute error (real number).

    """    
    tot_err = 0.0
    for (out, tar) in details:
        tot_err += norm(tar - out)
    mean_err = tot_err / len(details)
    return mean_err

def residual_errors(details):
    """Return residual errors as a list of scipy.arrays shaped (|D|, Ny)"""
    errors = scipy.array([tar - out for (out, tar) in details])
    assert errors.shape == (len(details), details[0][0].shape[0]), \
        "Wrong shape: %s" % errors.shape
    return errors

def normalized_residual_errors(details, sup=0.9):
    """Return the residual errors after been scaled to fit 
    the interval [-sup, +sup] on each dimension.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    sup     --  Positive value of the final interval [-sup, +sup] (real).

    Return:
    Normalized residual errors as a scipy.array shaped (|D|, Ny) with values
    in [-sup, +sup].

    """
    err = [tar - out for (out, tar) in details]
    Ny = err[0].shape[0]
    maxerr = [max([abs(d[idx]) for d in err]) for idx in range(Ny)] 
    errors = scipy.array(err) * sup / maxerr
    assert errors.shape == (len(details), Ny), "Wrong shape: %s" % errors.shape
    return errors

def scaled_residual_errors(details, scaling=0.5):
    """Return the residual errors after been scaled of a fized factor.

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    scaling --  Scaling factor (real number).

    Return:
    Scaled residual errors as a scipy.array shaped (|D|, Ny).

    """
    err = [tar - out for (out, tar) in details]
    errors = scipy.array(err) * scaling
    assert errors.shape == (len(details), err[0].shape[0]), \
        "Wrong shape: %s" % errors.shape
    return errors

def encoded_residual_errors(details, value=1.0, classes=None):
    """Return the residual errors represented by a 1-of-k bipolar encoding.
    This function is suited to work with classification tasks (where target and 
    output values are discrete).

    Arguments:
    details --  A list of pairs {(y(g), y_target(g)) | g in dataset}, where 
                y, y_target are scipy.array shaped (Ny,).
    value   --  Value to use in the bipolar encoding (real number).
    classes --  Specify which class have to be considered. If 'None' then 
                classes will be retrieved by the data.

    Return:
    Encoded residual errors as a scipy.array shaped (|D|,|C|), where C is
    the set of the different residual errors values/classes.
        
    """
    err = [tuple(tar - out) for (out, tar) in details]
    cl = list(sorted(set(err))) if classes is None else classes # classes
    errors = []
    for e in err:
        enc = [-value] * len(cl)
        enc[cl.index(e)] = +value
        errors.append(enc)
    errors = scipy.array(errors)
    assert errors.shape == (len(details), len(cl)), \
        "Wrong shape: %s" % errors.shape
    return errors
    

