"""Implementation of the constructive GraphESN model."""

import scipy
from g_esn.graph_esn import GraphESN


class PluggableGraphESN(GraphESN):
    """A single 'computational unit' in the constructive network.

    A PluggableGraphESN is a GraphESN accepting output feedbacks from other
    networks. New connections are allowed from other networks to the the
    reservoir or to the readout.

    Reservoir's output feedbacks have weights in a new matrix W_aux 
    shaped (Nr, Nz_res). Encoding process is modified in order to consider also
    output feedback values.

    Readout's output feedbacks are managed using the same W_out matrix, as in
    the standard GraphESN, reshaped as (Ny, Nr + Nz_ro + 1). Indeed these 
    output feedbacks are considered as additional features.
        
    Methods:
    --------
    join_to_reservoir   --  Add new output feedbacks to the reservoir.
    join_to_readout     --  Add new output feedbacks to the readout.
    encode              --  Overrided encoding method, using output feedbacks.
    get_activations     --  Overrided get_activation method. 


    Attributes:
    -----------
    W_aux           --  Reservoir's output feedbacks weight matrix 
                        (scipy.array (Nr, Nz_res)).
    Nz_res          --  Numeber of reservoir's output feedbacks.
    Nz_ro           --  Numeber of readout's output feedbacks.
    reservoir_sub   --  List of networks connected to the reservoir (list of
                        PluggableGraphESN)
    readout_sub     --  List of networks connected to the readout (list of
                        PluggableGraphESN)

    """

    def __init__(self, W_in, W, W_out, mapping_fun, 
            output_act=None, epsilon=10**-5, maxit=100):
        """Initialize new PluggableGraphESN with no output feedbacks.
        
        Arguments:
        W_in        --  Input-to-reservoir matrix. (scipy array (Nr,Nu+1))
        W           --  Reservoir-to-reservoir matrix (scipy.array (Nr,Nr))
        W_out       --  Reservoir-to-output matrix (scipy.array (Ny,Nr+1))
        mapping_fun --  State mapping function f:(Nr x |V(g)|) -> Nr
        output_act  --  Output unit's activation function. See: g_esn.functions
                        (callable)
        epsilon     --  Reservoir convergence threshold (real).
        maxit       --  Reservoir maximum iteration number (int).

        See also: g_esn.graph_esn.GraphESN

        """ 
        GraphESN.__init__(self, W_in, W, W_out, mapping_fun, output_act, 
            epsilon, maxit)
        self._W_aux = scipy.ones((self.Nr,0))
        self.reservoir_sub = []
        self.readout_sub = []

    # New properties: W_aux, Nz_res and Nz_ro

    @property 
    def W_aux(self):
        """Reservoir's output feedbacks matrix. (Nr, Nz_res)"""
        return self._W_aux
    @W_aux.setter
    def W_aux(self, val):
        self._clear_encoding_cache()
        self._W_aux = val
    @W_aux.deleter
    def W_aux(self):
        del self._W_aux

    @property
    def Nz_res(self): 
        """Reservoir's output feedbacks size."""
        return self._W_aux.shape[1]

    @property
    def Nz_ro(self): 
        """Readout's output feedbacks size."""
        return self._W_out.shape[1] - self.Nr - 1       

    def join_to_readout(self, gesn, weights=None):
        """Add new subnetwork and use its output as an auxiliary input for
        the readout.

        Output weights matrix W_out will be modified by adding one column
        for each new input. Weights referring to bias term will be "shifted" to 
        the last column.
        
            E.g. [W|bias] becomes [W|W_z|bias]

        Arguments:
        gesn    --  New subnetwork (GraphESN)
        weights --  New inputs weights (scipy.array (Ny, gesn.Ny))
                    [def: None, i.e. all weights 1.0]

        """
        if (not weights == None) and (not weights.shape == (self.Ny, gesn.Ny)):
            raise Exception("Wrong subnetwork weights shape: %s" % weights.shape)
        self.readout_sub.append(gesn)
        if weights == None : # default weights
            weights = scipy.ones((self.Ny, gesn.Ny))
        parts = (self.W_out[:,:-1], weights, self.W_out[:,-1:]) # [W|W_z|bias]
        self.W_out = scipy.concatenate(parts, 1)
        #self._clear_activation_cache() # don't forget to sweep the cache

    def join_to_reservoir(self, gesn, weights=None):
        """Add new subnetwork and use its output as an auxiliary input for
        the reservoir.

        Reservoir auxiliary matrix W_aux will be modified by adding one column
        for each new input. 
        
            E.g. [W] becomes [W|W_z]

        Arguments:
        gesn    --  New subnetwork (GraphESN)
        weights --  New inputs weights (scipy.array (Nr, gesn.Ny))
                    [def: None, i.e. all weights 1.0]

        """
        if (weights is not None) and (not weights.shape == (self.Nr, gesn.Ny)):
            raise Exception("Wrong subnetwork weights shape: %s" % weights.shape)
        self.reservoir_sub.append(gesn)
        if weights is None : # default weights
            weights = scipy.ones((self.Nr, gesn.Ny))
        self.W_aux = scipy.concatenate((self.W_aux, weights), 1)
    
    def encode(self, graph, input_label):
        """Override the GraphESN.encode method.

        New encode is performed considering also auxiliary reservoir inputs,
        possibly coming from other PluggableGraphESN.
        Encoding step il modified as following:

        x_t(v) = tanh( W_in u(v) + \sum_{w \in N(v)} W x_{t-1}(w) + W_aux z_aux(g) )

        It's worth mentioning that z_aux(g) is the concatenation of values all
        referring to the whole graph instead of the current vertex. This allows
        the local encoding process to exploit contextual informations.

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
            Wz = self._compute_Wz(graph, input_label) # W_aux z_sub(g), (Nr,)
            Wuz = Wu + Wz # W_in u(v) + W_sub z_sub(g) forall v, (|V|,Nr)            
            x = scipy.zeros((len(graph.vertices), self.Nr)) # (|V|,Nr)
            done = False
            it = 0 # iteration counter
            while (not done) and it < self.maxit:
                if it == 0: # x_0(v) = 0 = Wx_0 for each vertex
                    next_x = scipy.tanh(Wu)
                else:
                    Wx = self._compute_Wx(graph, x) # (|V|,Nr)
                    next_x = scipy.tanh(Wx + Wuz)
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

    def _compute_Wz(self, graph, input_label):
        """Compute ( W_sub z_sub(g) ) for given graph.

        Return: Result vector (Nr,)

        """
        if len(self.reservoir_sub) > 0 :
            z = [s.evaluate(graph, input_label) for s in self.reservoir_sub]
            zaux = scipy.concatenate(z)
            assert zaux.shape == (self.Nz_res,), \
                "Invalid shape zaux: %s" % str(zaux.shape)
            Wz = scipy.dot(self.W_aux, zaux)
            assert Wz.shape == (self.Nr,), \
                "Invalid shape Wz: %s" % str(Wz.shape)
        else: # there are no auxiliary inputs
            Wz = scipy.zeros((self.Nr,)) 
        return Wz

    def get_activations(self, graph, input_label):
        """Override the GraphESN.get_activations method by adding the capability
        to deal with auxiliary inputs for the readout.

        How it works:
            - Each auxiliary (readout) input is evaluated by calling the 
              evaluate method on the subnetworks. Subnetworks ordering is 
              preserved (i.e. first-joined first-evaluated)
            - Obtained auxiliary inputs, z_i, are concatenated into a single 
              vector (Nz_ro,).
            - Reservoir activations are evaluated normally, by calling the 
              'encode' method and applying the state mapping functions
            - Reservoir activations and auxiliary inputs are joined into a
              single vector, shaped ((Nr + Nz_ro),).

        Arguments:
        graph       --  Input graph (Graph)
        input_label --  Name of the attribute storing the input label.

        Return:
        Activations values, i.e. readout input. (scipy.array ((Nr + Nz_ro),)).    

        """
        # collect auxiliary inputs
        if len(self.readout_sub) > 0 :
            z = [s.evaluate(graph, input_label) for s in self.readout_sub]
            zaux = scipy.concatenate(z)
            assert zaux.shape == (self.Nz_ro,), \
                "Invalid shape zaux: %s" % str(zaux.shape)
        else: # no auxiliary inputs
            zaux = scipy.zeros((0,))
        # reservoir activations
        Xg = GraphESN.get_activations(self, graph, input_label)
        Xz = scipy.concatenate((Xg,zaux))
        return Xz            
