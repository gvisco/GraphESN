#!/bin/env python
"""
--------------------------------------------------------------------------------
                    Constructive GraphESN sample code
--------------------------------------------------------------------------------

Targets: 
1 - Load data from sample dataset 'dataset.sdf' and prepare the dataset.
2 - Split the dataset in two parts: training-set and test-set.
3 - Create a Constructive GraphESN with Nr=50
4 - Incrementally add 10 subnetworks performing both training and test

--------------------------------------------------------------------------------
"""

########## Step 0: some preliminary stuff

# import scipy to deal with matrices
import scipy
# import the class implementing a single computational unit (PluggableGraphESN)
from g_esn.constructive.pluggable_graph_esn import PluggableGraphESN
# import the parser for SDF data
from g_esn.parsers import sdf_parser
# import the mapping function (i.e. mean-state-mapping) and other functions
from g_esn.functions import mean_state_mapping, tanh, step_function
# some useful function to deal with matrices
from g_esn.utils import matrices
# learning algorithms: Ridge Regression (subnetwork) LMS (global)
from g_esn.learning import ridge_regression, lms
# from the 'dataset utilities package' a function to get the atomic symbols
from g_esn.utils.dataset import get_attribute_values
# a function to remove graphs with no target specified
from g_esn.utils.dataset import purge_dataset

########## Step 1: load the data

# parse data
path = 'dataset.sdf'
dataset = sdf_parser.parse(path) # 'dataset' is a list of Graph objects
# create labels with binary encoding
symbols = get_attribute_values(dataset, 'symbol', vertex=True) # get symbols
for v in [vertex for g in dataset for vertex in g.vertices]: # for each vertex
    lbl = [0.0] * 22 # size: 22 (only the binary encoding)
    lbl[symbols.index(v.attr['symbol'])] = 1.0 # 1-of-k representation
    v.attr['label'] = scipy.array(lbl) # Set 'label' as the input attribute
# take only graphs with target 'PTC.CLASS.FM'
dataset = filter(lambda g: 'PTC.CLASS.FM' in g.attr, dataset)

########## Step 2: split [------DATASET------] => [-----TR-----|--TS--]

test_size = 20              
trainset = dataset[:-test_size] # from 0 to |dataset| - test_size - 1
testset = dataset[-test_size:] # from |dataset| - test_size to the end
# get Targets as matrix
trTarget = scipy.array([[float(g.attr['PTC.CLASS.FM'])] for g in trainset])
tsTarget = scipy.array([[float(g.attr['PTC.CLASS.FM'])] for g in testset]) 

########## Step 3: create new (constructive) GraphESN
# 
# We will use a trick to obtain the global readout by using a PluggableGraphESN 
# with no reservoir at all (i.e. Nr=0).
# 

k = 4
Nu = 22 # input size
Ny = 1 # output size
Nr = 0 # the "zero-network" has no reservoir (i.e. only the global readout)
out_act = tanh # use tanh as output activation function
# create needed matrices (this doesn't really matter since Nr=0)
W_in = matrices.random((Nr, Nu+1))  # W_in (n.b. the bias term)
W = matrices.random((Nr, Nr)) # \hat{W}
W_out = matrices.random((Ny, Nr+1)) # W_out (n.b. the bias term)
network = PluggableGraphESN(W_in, W, W_out, mean_state_mapping, out_act)


########## Step 4: constructive loop

# set the first residual error as -Target
residuals = -trTarget
for sub_idx in range(1, 11): # perform 10 iterations [1,10]

    ### create new subnetwork
    # set some parameter
    Nr = 50 # (sub)reservoir size
    in_scaling = 1.0 # input scaling
    density = 1.0 # \hat{W} density
    sigma = 1.0 # contraction coefficient
    mappingfun = mean_state_mapping # mapping function
    out_act = tanh # use tanh as output activation function
    # create matrices
    W_in = matrices.random((Nr, Nu+1)) * in_scaling # W_in
    W = matrices.random((Nr, Nr)) # \hat{W}
    matrices.set_density(W, density) # set desired density
    W = matrices.contractive(W, sigma, k) # set contractive setting    
    W_out = matrices.random((Ny, Nr+1)) # W_out
    # create GraphESN    
    subnet = PluggableGraphESN(W_in, W, W_out, mappingfun, out_act)    

    ### connect to existing subnetworks (assuming FOF)
    for net in network.readout_sub: # networks connected to the global readout
        subnet.join_to_readout(net) # connect existing network to the readout
        subnet.join_to_reservoir(net)  # connect to the reservoir    

    ### train the subnetwork using Ridge Regression
    residuals = residuals * 0.5 # error in (-1.0,+1.0)
    rlambda = 0.0 # regularization parameter for Ridge Regression
    ridge_regression(subnet, trainset, 'label', residuals, rlambda)

    ### add the subnetwork to the global readout
    network.join_to_readout(subnet)

    ### train the global readout using LMS
    eta = 10**-4
    threshold = 10**-3
    lms(network, trainset, 'label', trTarget, eta)    
    # get the output 
    trOut = scipy.array([network.evaluate(g, 'label') for g in trainset]) 
    tsOut = scipy.array([network.evaluate(g, 'label') for g in testset]) 

    ### get the misclassification-rate
    output_function = step_function(1.0)
    # tr
    tr_errors = trTarget - output_function(trOut)
    total_err = scipy.sum([scipy.any(tr_errors != 0.0, 1)])
    tr_misclass = total_err / float(tr_errors.shape[0]) # misclassification rate
    # ts
    ts_errors = tsTarget - output_function(tsOut)
    total_err = scipy.sum([scipy.any(ts_errors != 0.0, 1)])
    ts_misclass = total_err / float(ts_errors.shape[0]) # misclassification rate
    print "(%2i) TR:%0.2f \tTS:%0.2f" % (sub_idx, tr_misclass, ts_misclass)

    ### update the residual errors
    residuals = trOut - trTarget

    
    

