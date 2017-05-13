""" Implement basic functionalities of GraphESN. 

Modules:
graph       --  A simple directed/undirected graph with attributes and 
                labeled vertices.
graph_esn   --  A standard GraphESN used to perform structure-to-element 
                mappings.

Subpackages:
functions       --  Common functions implementation (e.g. unit's activation 
                    function).
learning        --  Learning algorithms used to train the networks.
constructive    --  Constructive extensions of the GraphESN model.
parsers         --  Read files into labeled graph objects.
utils           --  Collection of utilities.

"""

__all__ = [
    "graph",        # Directed Graph (base data in a dataset) 
    "graph_esn"     # Standard GraphESN
    ]
