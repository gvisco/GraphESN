"""
Module provides a Python representation of a GPH-standard file. Each graph is
parsed into a graph.Graph object.

In order to "enjoy" using a GraphESN the trees represented by GPH file will
be saved as undirected graphs.

See below - at the end of the source file - for more informations about GPH 
standard file format.

"""

import re
from g_esn.graph import Graph

DEB = False

class GPHParserError(Exception):
    """ Errors related to GPH format.  """ 
    def __init__(self, value): self.value = value
    def __str__(self): return repr(self.value)

currline = 0 # last read line counter

def parse(path):
    global currline
    # reset some data
    currline = 0
    # read file 
    fo = open(path, 'r')
    dataset = []
    t = __next_tokens(fo)
    # preamble
    sym_num = get_key_val(t, 'SymbolNum', int)
    t = __next_tokens(fo)
    label_dim = get_key_val(t, 'LabelDim', int)
    # symbol table
    sym_table = __parse_symtable(t, fo, sym_num, label_dim)
    # header
    t = __next_tokens(fo)
    tree_num = get_key_val(t, 'TreeNum', int)
    t = __next_tokens(fo)
    max_arity = get_key_val(t, 'MaxArity', int)
    # data
    for counter in xrange(tree_num):
        t = __next_tokens(fo)
        tree_name = get_key_val(t, 'Name', str)       
        t = __next_tokens(fo)
        tree_dim = get_key_val(t, 'TreeDim', int)
        t = __next_tokens(fo)
        tree_target = get_key_val(t, 'Target', float) # tree-to-element task
        # create new Graph object (vertices will be modified later)
        # XXX: assuming mono-dimensional target
        tree_attr = {"target":[tree_target], 'name':tree_name, 'dimension':tree_dim}
        g = Graph(directed=False, attr=tree_attr, N=tree_dim)
        # connection table
        for _ in xrange(tree_dim):
            t = __next_tokens(fo)
            v_id = int(t[0]) 
            v_edges = map(int, t[1 : 1 + max_arity])
            v_sym_name = t[1 + max_arity]
            v_sym_id = int(t[2 + max_arity])
            v_parent_id = int(t[3 + max_arity])
            # modify vertex. XXX Assuming the id is the index
            g.vertices[v_id].attr['symbol'] = v_sym_name 
            g.vertices[v_id].attr['symbol_id'] = v_sym_id
            g.vertices[v_id].attr['parent'] = v_parent_id
            # add edges. XXX ignore position
            v_edges = filter(lambda x : not x == -1, v_edges) 
            for dest in v_edges: 
                g.add_edge(v_id, dest)
        dataset.append(g)     
    fo.close() # close input file
    return dataset


def __parse_symtable(t, fo, sym_num, label_dim):
    """Parse the Symbol Table in a dictionary.

    Return: 
    {symbol_id : (name(string), outdeg(int), label(list of float))} 

    """
    table = {} # indexed by symbol_id
    for _ in xrange(sym_num):
        t = __next_tokens(fo)
        s_id = int(t[0]) 
        s_name = t[1]
        s_outdeg = int(t[2])
        s_label = map(float, t[3:]) # actually label_dim is not necessary
        table[s_id] = (s_name, s_outdeg, s_label)
    return table        

def get_key_val(tokens, key, vtype):
    """Get a couple of tokens. Verify teh first token to be equals to given key
    then return the value, casted using vtype.
    If the the key is invalid a GPHParserError is raised.

    """
    if tokens[0] == key : 
        return vtype(tokens[1])
    else :
        raise GPHParserError("(%i) Expecting '%s <value>' found: %s" % (currline, key, " ".join(tokens)))
    

def __next_tokens(fo):
    """Read next (unempty) line and return its tokens."""
    global currline
    while True :
        line = fo.readline();
        currline += 1
        if not "\n" in line : # end of file
            return None
        else:
            line = line.strip()
            if line != "" and line[0] != "#" : # else: skip
                return re.split(r"\s+", line) # remove tabs and spaces

def __deb(txt):
    global DEB
    if DEB : print txt


"""####### GPH Standard format #######

-- Preamble --

SymbolNum <num> # number of symbols (atoms)
LabelDim <num> # label dimension

"Symbol Table": each row is composed by
<id> # integer identifying a symbol 
<name> # string (atomic symbol)
<outDegree> # outdegree (in molecule domain each atom has fixed outdegree)
<vectorialLabel> # 'LabelDim' label values

e.g.

 0  c      3      1
 1  ch     2      1
 2  ch2    1      1
 3  ch3    0      1
 4  ch3f   1      1
 5  ch4    0      1


-- Header --

TreeNum <num> # total number of trees in the dataset
MaxArity <num> # maximum outdegree 


-- Data -- 

Name <string> # tree id 
TreeDim <num> # number of vertices
Target <num_real> # target value for given task (assuming tree-to-element task)

"Connection Table": each row is composed by
<vertexID> # vertex id 
<edges>+ # 'MaxArity' connections to other vertices (-1 for 'nil')
<symbolName> # symbol name, see Symbol Table above
<symbolID> # symbol id, see Symbol Table above 
<parentID> # parent id (-1 for 'nil', e.g. the root vertex)

e.g.

 0   -1 -1 -1  ch3        3     1
 1   -1 -1  0  ch2        2     2
 2   -1 -1  1  ch3f       4    -1

"""
