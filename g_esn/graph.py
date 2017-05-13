class GraphError(Exception):
    """Errors related to Graph creation or manupulation"""
    def __init__(self, value): self.value = value
    def __str__(self): return repr(self.value)


class Vertex(object):
    """A Vertex.

    Each vertex has attributes stored in the dictionary 'attr' and
    stores outgoing edges ('out_conn') and incoming edges ('in_conn') as lists 
    of integers, indicating the intex of an other vertex in the graph.

    Attributes:
    attr        --  Vertex attributes (dictionary).
    out_conn    --  Indices of vertices pointed by this vertex (list of int).
    int_conn    --  Indices of vertices pointing to this vertex (list of int).

    """
    def __init__(self, attr=None):
        """Initialize new Vertex.

        Arguments:
        attr    --  Vertex attributes (dictionary)[def: None].

        """        
        self.out_conn = []
        self.in_conn = []
        self.attr = {} if attr is None else attr  # e.g. "target" 


class Graph(object):
    """A (un)directed graph.

    Each Graph may have its own attributes stored in the 'attr' dictionary.
    Informations about edges are stored by vertices. Integers are used to 
    identify a vertex and then to define new edges.

    Undirected Graphs are implemented as directed graphs having edges in both 
    directions.

    Vertices and edges can be added.

    Attributes:
    directed    --  Tells whether the graph is directed or not (boolean).
    attr        --  Graphs attributes (dictionary).
    vertices    --  Graph vertices (list of Vertex).

    """
    def __init__(self, directed=False, attr=None, N=0):
        """Create new Graph.

        Arguments:
        directed    --  Tells whether the graph is directed [def: False].
        attr        --  Graph attributes (dictionary) [def: {}]
        N           --  Initial number of vertices (int) [def: 0]

        """
        self.directed = directed
        self.attr ={} if attr is None else attr # e.g. "target" 
        self.vertices = [Vertex() for _ in xrange(N)]

    def add_vertex(self, v=None):
        """Add new vertex and return its index.

        If no Vertex is specified then new 'empty' Vertex will be created.

        Arguments:
        v   --  Vertex to be added (Vertex) [def: None].

        Return:
        Index of the added vertex.

        """
        if v is None : v = Vertex()
        self.vertices.append(v)
        return len(self.vertices) - 1

    def add_edge(self, vidx1, vidx2, multi_edge=False):
        """Add new edge.

        If the graph is not directed then two edges will be added (in both 
        directions).

        Arguments:
        vidx1       --  Index of the source vertex.
        vidx2       --  Index of the destination vertex.
        multi_edge  --  Allow or disallow more than one edge between two 
                        vertices (boolean) [def: False].

        Errors:
        GraphError  --  Raised when indices are out of bounds.

        """
        # check indices
        if vidx1 >= len(self.vertices):
            raise GraphError("Adding new edge. Index out of bounds: %i" % vidx1)
        if vidx2 >= len(self.vertices):
            raise GraphError("Adding new edge. Index out of bounds: %i" % vidx2)
        # check edges
        if not multi_edge and vidx2 in self.vertices[vidx1].out_conn :
            raise GraphError("Duplicated edge: %i -> %i" % (vidx1, vidx2))
        if not multi_edge and vidx1 in self.vertices[vidx2].in_conn :
            raise GraphError("Inconsistent edge status. Vertices: %i and %i" % (vidx1, vidx2))
        # add edge
        self.vertices[vidx1].out_conn.append(vidx2)
        self.vertices[vidx2].in_conn.append(vidx1)
        if not self.directed :
            # check 'backward' edge
            if not multi_edge and vidx1 in self.vertices[vidx2].out_conn :
                raise GraphError("Duplicated edge: %i -> %i" % (vidx2, vidx1))
            if not multi_edge and vidx2 in self.vertices[vidx1].in_conn :
                raise GraphError("Inconsistent edge status. Vertices: %i and %i" % (vidx2, vidx1))
            # add edge
            self.vertices[vidx2].out_conn.append(vidx1)
            self.vertices[vidx1].in_conn.append(vidx2)
