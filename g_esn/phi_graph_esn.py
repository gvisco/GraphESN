"""Phi-GraphESN implementation."""
import scipy
from g_esn.graph_esn import GraphESN

class PhiGraphESN(GraphESN):
    """Phi-GraphESN implementation.

    Extend a standard GraphESN by adding a random weighted layer that expand the 
    reservoir's output in order to get a higher dimensional feature space.

    """

    def __init__(self, W_in, W, W_phi, W_out, mapping_fun, 
            output_act, epsilon=10**-5, maxit=100):
        """ 

        Arguments:
        W_in        --  Input-to-reservoir matrix (scipy.array (Nr, Nu+1)).
        W           --  Reservoir-to-reservoir matrix (scipy.array (Nr, Nr)).
        W_phi       --  Expansion matrix (scipy.array (Nphi, Nr))
        W_out       --  Expansion-to-readout matrix (scipy.array (Ny, Nphi+1))
        mapping_fun --  State mapping function f:(Nr x |V(g)|) -> Nr
        output_act  --  Output unit's activation function.
        epsilon     --  Reservoir convergence threshold (real).
        maxit       --  Reservoir iteration limit (int).

        See also: g_esn.graph_esn.GraphESN

        """
        GraphESN.__init__(self, W_in, W, W_out, mapping_fun, output_act, 
            epsilon, maxit)
        self.W_phi = W_phi
        
    def check(self):
        """Verify if internal matrices are well shaped.
        
        Return a boolean value.

        """
        if not self.W_in.shape == (self.Nr, self.Nu + 1):
            return False 
        elif not self.W.shape == (self.Nr, self.Nr):
            return False 
        elif not self.W_phi.shape == (self.Nphi, self.Nr):
            return False 
        elif not self.W_out.shape == (self.Ny, self.Nphi + 1):
            return False 
        else:
            return True

    @property 
    def W_phi(self):
        """Reservoir expansion matrix (dynamic to static). (Nphi, Nr)"""
        return self._W_phi
    @W_phi.setter
    def W_phi(self, val):
        self._clear_activation_cache()
        self._W_phi = val
    @W_phi.deleter
    def W_phi(self):
        del self._W_phi

    @property
    def Nphi(self): 
        """Static reservoir part size"""
        return self._W_phi.shape[0]

    def get_activations(self, graph, input_label):
        """Return the encoding for given graph.

        Expand the result of the state mapping function using W_phi matrix.

        Arguments:
        graph       --  Input graph (g_esn.graph.Graph)
        input_label --  Name of the attribute storing the input label (string)

        Return:
        Encoding in the expanded feature space (scipy.array (Nphi,)).

        """
        x = GraphESN.get_activations(self, graph, input_label)
        x = scipy.dot(self.W_phi, x) # phi expansion (Nphi, Nr) x (Nr,)
        return scipy.tanh(x)


