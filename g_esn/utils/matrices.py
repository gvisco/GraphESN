"""Provide some useful function to Ease the manipulation of bidimensional 
scipi.array objects (matrices from the GraphESN perspective).

"""

import random as rnd

import scipy
import scipy.linalg as linalg

def from_file(path, sep=' ', data_type=scipy.float64):
    """Read a matrix from a file.
    
    Arguments:
    path        --  Path of the file to read (string).
    sep         --  Column separator (string) 
    data_type   --  Output array data type.

    Return:
    Read matrix as scipy.array.    

    """
    matrix = []
    for line in [l.strip() for l in open(path, 'r')]:
        matrix.append( map(data_type, line.rsplit(sep)) )
    return scipy.array(matrix)

def print_matrix(m, sep=' '):
    """Print given matrix.
    
    Each value will be printed by using 'repr' function, in order
    to get more decimal values than a simple print.

    Arguments:
    m   --  Matrix to print.
    sep --  Column separator.

    """
    for row in [sep.join(map(repr, r)) for r in m]:
        print row

def to_string(m, sep=' '):
    """Return given matrix as string.

    Each value will be obtained by using 'repr' function, in order
    to get more decimal values than a simple print.

    Arguments:
    m   --  Matrix to print.
    sep --  Column separator.

    Return:
    A string representation of the matrix.

    """
    s = ''
    for row in [sep.join(map(repr, r)) for r in m]:
        s += "%s\n" % row
    return s

def random(shape, (minval, maxval)=(-1.0,+1.0)):
    """Return new matrix containing randomly generated uniformly
    distributed values.
    
    Arguments:
    shape               --  Matrix shape (tuple of int)
    (minval, maxval)    --  Range values.
    
    Return: 
    New matrix (scipy.array)

    """    
    return scipy.random.uniform(minval, maxval, shape)

def contractive(matrix, sigma, k):
    """Scale given matrix to force contractivity.

    How it works:
        Contraction coefficient is defined as:

            \sigma = ||matrix||_2 k

        this function returns a matrix with contraction coefficient
        equals to \sigma by calculating:

            matrix' = (\sigma / ||matrix||_2 / k) * matrix

    Arguments:
    matrix  --  Matrix to scale (scipy.array).
    sigma   --  Contractivity factor (real number).
    k       --  Maximum degree in the dataset (int number).

    Return:
    The scaled matrix (scipy.array).

    """
    return (sigma / linalg.norm(matrix) / k) * matrix

def set_density(matrix, density):
    """Remove random entries (set them to '0.0') from given matrix until desired 
    density is reached.
    If original density is lower than desired density then nothing will be done.
    
    Arguments:
    matrix  --  Matrix to change (scipy.array)
    density --  Rate of non-zero values (real number in [0,1])

    """
    size = matrix.size
    if size == 0:
        return
    candidates = []
    zeros = 0
    for ir, row in enumerate(matrix):
        for ic, value in enumerate(row):
            if value == 0.0 :
                zeros += 1
            else:
                candidates.append( (ir,ic) )
    rnd.shuffle(candidates)
    currd = 1 - zeros / float(size)
    while currd > density:
        (ridx, cidx) = candidates.pop()
        matrix[ridx, cidx] = 0.0
        zeros += 1
        currd = 1 - zeros / float(size)

def get_density(matrix):
    """Get the the density of given matrix.

    Arguments:
    matrix  --  Matrix to check (scipy.array)

    Return:
    Matrix density (real value).

    """
    zeros = 0
    for r in matrix:
        for val in r:
            if val == 0.0 : zeros += 1
    return 1 - (zeros / float(matrix.size))
