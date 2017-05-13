"""Implementation of the constructive PhiGraphESN model."""

import scipy
from g_esn.graph_esn import GraphESN
from g_esn.phi_graph_esn import PhiGraphESN
from g_esn.constructive.pluggable_graph_esn import PluggableGraphESN

class PluggablePhiGraphESN(PluggableGraphESN, PhiGraphESN):
    """PluggablePhiGraphESN extends both PluggableGraphESN and PhiGraphESN in 
    order to obtain a computational unit suited for constructive networks that
    also uses a static expansion layer as a PhiGraphESN.

    """

    def __init__(self, W_in, W, W_phi, W_out, mapping_fun, 
            output_act=None, epsilon=10**-5, maxit=100):
        """Initialize new PluggablePhiGraphESN with no output feedbacks.
        
        Arguments:
        W_in        --  Input-to-reservoir matrix. (scipy array (Nr,Nu+1))
        W           --  Reservoir-to-reservoir matrix (scipy.array (Nr,Nr))
        W_phi       --  Expansion layer weights (scipy.array (Nphi, Nr))
        W_out       --  Reservoir-to-output matrix (scipy.array (Ny,Nphi+1))
        mapping_fun --  State mapping function f:(Nr x |V(g)|) -> Nr
        output_act  --  Output unit's activation function. See: g_esn.functions
                        (callable)
        epsilon     --  Reservoir convergence threshold (real).
        maxit       --  Reservoir maximum iteration number (int).

        See also: PluggableGraphESN and PhiGraphESN.

        """ 
        PluggableGraphESN.__init__(self, W_in, W, W_out, mapping_fun, 
            output_act, epsilon, maxit)
        PhiGraphESN.__init__(self, W_in, W, W_phi, W_out, mapping_fun, 
            output_act, epsilon, maxit)
    
    @property
    def Nz_ro(self): 
        """Readout's auxiliary inputs size."""
        return self._W_out.shape[1] - self.Nphi - 1     

    def get_activations(self, graph, input_label):
        """Return the encoding for given graph considering also reservoir's 
        output feedback (if any).

        Expand the result of the state mapping function using W_phi matrix.

        Arguments:
        graph       --  Input graph (g_esn.graph.Graph)
        input_label --  Name of the attribute storing the input label (string)

        Return:
        Encoding in the expanded feature space (scipy.array (Nphi,)).

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
        key = self._extract_key(graph, input_label)
        if key in self._cache_activations:
            Xg = self._cache_activations[key]
        else:
            xgs = self.encode(graph, input_label) # x(g)
            Xg = self.map_fun(xgs)
            # phi expansion
            Xg = scipy.dot(self.W_phi, Xg) # (Nphi, Nr) x (Nr,) = (Nphi)
            Xg = scipy.tanh(Xg) # apply non-linearity
            if self.memo_act:
                self._cache_activations[key] = Xg
        Xz = scipy.concatenate((Xg,zaux))
        return Xz

