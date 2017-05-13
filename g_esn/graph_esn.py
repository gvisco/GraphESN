"""Module defines the class GraphESN, implementing the standard 
Echo State Network for graphs.

"""

import scipy

class GraphESN(object):
    """Standard GraphESN implementation.

    A GraphESN performs following operations:

        Input -> Encoding -> Mapping -> Readout -> Output Activation

    See details below.

    GraphESN can use memoization, avoiding the reprocessing of previously
    evaluated inputs.

    Attributes:
    -----------
    Nu      --  Input label dimension. (int)
    Nr      --  Reservoir size, i.e. number of units. (int)
    Ny      --  Output size. (int)
    W_in    --  Input-to-reservoir matrix (scipy.array (Nr,Nu+1))
    W       --  Reservoir-to-reservoir matrix (scipy.array (Nr,Nr))
    W_out   --  Reservoir-to-readout matrix (scipy.array (Ny,Nr+1))
    map_fun --  State mapping function: (|V(g)|,Nr) -> (Nr,)
    out_act --  Output activation function: (Ny,) -> (Ny,)
    epsilon --  Reservoir convergence threshold (float).
    maxit   --  Maximum number of iterations allowed for the reservoir (int).


    Main methods:
    -------------
    encode          --  Get the internal representation of a Graph.
                        (list of scipy.array (Nr,))
    get_activations --  Get the reservoir activations after the state mapping
                        function has been applied. (scipy.array (Nr,))
    evaluate        --  Get the output for an input graph. (scipy.array (Ny,))


    Memoizing:
    ----------
    memo_enc    --  Save results for the 'encode' function (boolean).
    memo_act    --  Save results for the 'get_activations' function (boolean).
    memo_eval   --  Save results for the 'evaluate' function (boolean).

    All the values are True by default.

    ----------------------------------------------------------------------------

    Input
    ------------------------
    A (directed) Graph 'g = (V(g), E(g))'. Each vertex has an attached label
    (scipy.array) in its attributes.


    Input -> Encoding 
    ------------------------
    Input graph is encoded into a vector. Internal representation is

        x(g) = scipy.array (Nr * |V(g)|,). 

    See method 'encode'.


    Encoding -> Mapping
    ------------------------
    Obtained encoding is mapped into a fixed length vector in order to
    achieve a size independent representation.
    A state mapping function, X : (Nr * |V(g)|,) -> (Nr,) is applied 
    e.g. 'mean_state_mapping' and 'supersource_state_mapping'.
    See method 'get_activations'.


    Mapping -> Readout
    ------------------------
    Reservoir activations returned by the mapping function are sent to network's
    readout: a linear combination is computed. Weights are stored in the
    matrix W_out (scipy.array (Ny, (Nr + 1)))

        net = W_out * X(x(g)), scipy.array (Ny,)
    

    Readout -> Output Activation
    ------------------------
    Linear combination result is used as input of the output activation 
    function, which is applied elementwise.

        out = f(net) = f(W_out * X(x(g)))

    In order to train the network, the output activation function can be  
    assumed to be derivable (e.g. 'g_esn.functions.tanh'). Thus it's recommended
    to use an ActivationFunction object (see package g_esn.functions).
    Output activation is the actual output of the network.
    See method 'evaluate'.    

    """

    def __init__(self, W_in, W, W_out, mapping_fun, 
            output_act, epsilon=10**-5, maxit=100):
        """Initialize new Graph-ESN.

        W_in and W_out are expected to have an additional column
        to deal with the bias term.
        
        Arguments:
        W_in        --  Input-to-reservoir matrix. (scipy array (Nr,Nu+1))
        W           --  Reservoir-to-reservoir matrix (scipy.array (Nr,Nr))
        W_out       --  Reservoir-to-output matrix (scipy.array (Ny,Nr+1))
        mapping_fun --  State mapping function f:(Nr x |V(g)|) -> Nr
        output_act  --  Output unit's activation function. See: g_esn.functions
                        (callable)
        epsilon     --  Reservoir convergence threshold (real).
        maxit       --  Reservoir maximum iteration number (int).

        """        
        self._W_in = W_in
        self._W = W
        self._W_out = W_out
        self._map_fun = mapping_fun
        self._out_act = output_act

        self._epsilon = epsilon
        self._maxit = maxit
        # memoizing
        self._memo_enc = True
        self._memo_act = True
        self._memo_eval = True

        self._cache_encode = {}
        self._cache_activations = {}
        self._cache_evaluate = {}

    def check(self):
        """Verify if internal matrices are well shaped.
        
        Return a boolean value.

        """
        if not self.W_in.shape == (self.Nr, self.Nu + 1):
            return False # Wrong W_in shape
        elif not self.W.shape == (self.Nr, self.Nr):
            return False # Wrong W shape
        elif not self.W_out.shape == (self.Ny, self.Nr + 1):
            return False # Wrong W_out shape
        else:
            return True


    # functions used for cache sweeping

    def _clear_encoding_cache(self):
        """Clear encoding + activation + evaluation caches."""
        self._cache_encode.clear()
        self._clear_activation_cache()
        self._clear_evaluation_cache()

    def _clear_activation_cache(self):
        """Clear activation + evaluation caches."""    
        self._cache_activations.clear()
        self._clear_evaluation_cache()

    def _clear_evaluation_cache(self):
        """Clear evaluation cache."""
        self._cache_evaluate.clear()

    # define properties (mostly needed to implement memoization)

    @property 
    def W_in(self):
        """Input to reservoir matrix. (Nr, Nu+1)"""
        return self._W_in
    @W_in.setter
    def W_in(self, val):
        self._clear_encoding_cache()
        self._W_in = val
    @W_in.deleter
    def W_in(self):
        del self._W_in

    @property 
    def W(self):
        """Reservoir to reservoir matrix. (Nr, Nr)"""
        return self._W
    @W.setter
    def W(self, val):
        self._clear_encoding_cache()
        self._W = val
    @W.deleter
    def W(self):
        del self._W

    @property 
    def W_out(self):
        """Reservoir to output matrix. (Ny, Nr+1)"""
        return self._W_out
    @W_out.setter
    def W_out(self, val):
        self._clear_evaluation_cache()      
        self._W_out = val
    @W_out.deleter
    def W_out(self):
        del self._W_out

    @property 
    def map_fun(self):
        """State Mapping Function."""
        return self._map_fun
    @map_fun.setter
    def map_fun(self, val):
        self._clear_activation_cache()
        self._map_fun = val
    @map_fun.deleter
    def map_fun(self):
        del self._map_fun

    @property 
    def out_act(self):
        """Output Activation Function."""
        return self._out_act
    @out_act.setter
    def out_act(self, val):
        self._clear_evaluation_cache()
        self._out_act = val
    @out_act.deleter
    def out_act(self):
        del self._out_act

    @property
    def epsilon(self):
        """Reservoir convergence threshold."""
        return self._epsilon
    @epsilon.setter
    def epsilon(self, val):
        self._clear_encoding_cache()
        self._epsilon = val
    @epsilon.deleter
    def epsilon(self):
        del self._epsilon

    @property
    def maxit(self):
        """Maximum reservoir iterations."""
        return self._maxit
    @maxit.setter
    def maxit(self, val):
        self._clear_encoding_cache()
        self._maxit = val
    @maxit.deleter
    def maxit(self):
        del self._maxit

    @property
    def memo_enc(self):
        """Use of memoizing for the 'encode' function"""
        return self._memo_enc
    @memo_enc.setter
    def memo_enc(self, val):
        if not val:
            self._clear_encoding_cache()
        self._memo_enc = val
    @memo_enc.deleter
    def memo_enc(self):
        del self._memo_enc

    @property
    def memo_act(self):
        """Use of memoizing for the 'get_activations' function"""
        return self._memo_act
    @memo_act.setter
    def memo_act(self, val):
        if not val:
            self._clear_activation_cache()
        self._memo_act = val
    @memo_act.deleter
    def memo_act(self):
        del self._memo_act

    @property
    def memo_eval(self):
        """Use of memoizing for the 'evaluate' function"""
        return self._memo_eval
    @memo_eval.setter
    def memo_eval(self, val):
        if not val:
            self._clear_evaluation_cache()
        self._memo_eval = val
    @memo_eval.deleter
    def memo_eval(self):
        del self._memo_eval


    # define Nu, Nr an Ny as properties (avoid assignement)

    @property
    def Nu(self): 
        """Input size"""
        return self._W_in.shape[1] - 1

    @property
    def Nr(self): 
        """Reservoir size"""
        return self._W.shape[0]

    @property
    def Ny(self): 
        """Output size"""
        return self._W_out.shape[0]
   
    def encode(self, graph, input_label):
        """Perform the encoding of given graph. 
        Return the internal state representation x \in (|V(g)| x Nr) as a 
        list of arrays.
        
        How it works:
            - Start from x=0.
            - For each vertex v compute: 
                x_t(v) = tanh( W_in u(v) + \sum_{w \in N(v)} W x_{t-1}(w) ).
            - Stop when 
                ||x_t - x_{t-1}|| < epsilon 
              for each vertex or after a fixed number of iterations.

        Note: no bias is needed in the input label. A bias term (+1) is added 
        during the encoding process.

        Arguments:
        graph       --  Input Graph.
        input_label --  Name of the attribute storing the input label.
                        Input label must be a scipy.array shaped (Nu,). (string) 
        
        Return:
        Internal representation of the graph (scipy.array (|V(g)|,Nr)).
        
        Exceptions:
        Raise an error if the encoding doesn't converge (i.e. after 'maxit'
        iterations).

        """
        key = self._extract_key(graph, input_label)
        if key in self._cache_encode:
            return self._cache_encode[key]
        else:
            Wu = self._compute_Wu(graph, input_label) # W_in u(v), (|V|,Nr)        
            x = scipy.zeros((len(graph.vertices), self.Nr)) # (|V|,Nr)
            done = False
            it = 0 # iteration counter
            while (not done) and it < self.maxit:
                if it == 0: # x_0(v) = 0 = Wx_0 for each vertex
                    next_x = scipy.tanh(Wu)
                else:
                    Wx = self._compute_Wx(graph, x) # (|V|,Nr)
                    next_x = scipy.tanh(Wx + Wu)
                # check for convergence
                done = self._check_convergence(x, next_x)
                x = next_x # update current encoding
                it += 1
            if it == self.maxit:
                raise Exception("Unable to reach reservoir convergence")
            assert x.shape == (len(graph.vertices), self.Nr), \
                "Invalid shape x(g): %s" % str(x.shape)
            if self.memo_enc:
                self._cache_encode[key] = x # save result
            return x

    def _check_convergence(self, curr_x, next_x):
        """Check if the encoding can be stopped.

        Encoding is stopped when for each vertex v in V(g):
    
            || x_{t}(v) - x_{t-1}(v) || <= epsilon

        Return: True if the encoding can be stopped. (boolean)

        """
        X = next_x - curr_x # (|V|,Nr)
        v = scipy.dot(X, X.T).diagonal() # (|V|,). v_i = \sum_j x_ij^2
        return scipy.all(v <= self.epsilon**2)

    def _compute_Wu(self, graph, input_label):
        """Compute ( W_in u(v) ) for each vertex 'v' in the graph.
        A bias is added to input label.

        Return: Obtained matrix (|V|,Nr)

        """
        U = scipy.array([v.attr[input_label] for v in graph.vertices])
        U = scipy.concatenate((U, scipy.ones((U.shape[0],1))), 1) # add bias
        Wu = scipy.dot(self.W_in, U.T).T
        assert Wu.shape == (len(graph.vertices), self.Nr), \
            "Invalid shape Wu: %s" % str(Wu.shape)
        return Wu
        
    def _compute_Wx(self, graph, x):
        """Compute ( \sum_{w \in N(v)} \hat{W} x_{t-1}(w) ) for each 
        vertex 'v' in the graph.

        Arguments:
        graph   --  Input graph (g_esn.graph.Graph)
        x       --  Current internal representation, i.e. x_{t-1} (|V|,Nr)

        Return: Obtained matrix (|V|,Nr)
 
        """
        sum_Wx = []
        Wx = scipy.dot(x, self.W.T) # (|V|, Nr)
        for (vidx, v) in enumerate(graph.vertices):
            sum_Wx_v = scipy.zeros((self.Nr,)) # save here the sum
            # XXX
            # if the graph is directed then ONLY SUCCESSORS are considered
            # while if the graph is undirected we are taking in account the 
            # whole neighborhood
            Nv = v.out_conn # neighbors
            for w in Nv:
                sum_Wx_v += Wx[w]
            sum_Wx.append(sum_Wx_v)
        sum_Wx = scipy.array(sum_Wx)
        assert sum_Wx.shape == (len(graph.vertices), self.Nr), \
            "Invalid shape Wx: %s" % str(sum_Wx.shape)
        return sum_Wx

    def get_activations(self, graph, input_label):
        """Let given graph to run thru the reservoir and return computed 
        activations.

        Returned encoding is evaluated after the state-mapping-function has 
        been applied so are intented to be the input for the readout.

        Arguments:
        graph       --  Input graph (Graph)
        input_label --  Name of the attribute storing the input label.

        Return: 
        Activations values (scipy.array (Nr,)).

        """
        key = self._extract_key(graph, input_label)
        if key in self._cache_activations:
            return self._cache_activations[key]
        else:
            xgs = self.encode(graph, input_label) # x(g)
            Xg = self.map_fun(xgs)
            if self.memo_act:
                self._cache_activations[key] = Xg
            return Xg # apply the state mapping function

    def evaluate(self, graph, input_label):
        """Apply the model to given graph. Return evaluated output value.
        
        Arguments:
        graph       --  Input graph (Graph)
        input_label --  Name of the attribute storing the input label.

        Return: 
        Real output values (scipy array (Ny,)).
        
        """
        key = self._extract_key(graph, input_label)
        if key in self._cache_evaluate:
            return self._cache_evaluate[key]
        else:       
            X = self._add_bias(self.get_activations(graph, input_label))
            net = scipy.dot(self.W_out, X)
            y = self.out_act(net)
            assert y.shape == (self.Ny,), "Wrong shape: output"
            if self.memo_eval:            
                self._cache_evaluate[key] = y # save result
            return y

    def _add_bias(self, v, val=1.0):
        """Concatenate a bias term to given vector.

        Arguments:
        v   --  Input vector (Scipy.array)
        val --  Bias value (float)

        Return:
        New modified vector.

        """
        return scipy.concatenate((v, [val]))

    def _extract_key(self, *args, **kwargs):
        """Return a keyword identifying given parameters (both named or not).

        This is a support function to implement the memoizing procedure.

        Return:
        A tuple that can be used as dictionary keyword.        

        """
        # An alternative and 'deeper' version using cPickle:
        #
        # import cPickle
        # return cPickle.dumps((args, kwargs))
        return args + tuple(sorted(kwargs.items()))
        
