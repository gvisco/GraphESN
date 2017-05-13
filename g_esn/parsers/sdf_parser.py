import re
from g_esn.graph import *

deb = False

class SDFParserError(Exception):
    def __init__(self, value): self.value = value
    def __str__(self): return repr(self.value)

currline = 0 # last read line counter

def parse(path, debug=False):
    """Parse a file and return read dataset.
    Add to each vertex a 'label' attribute used in machine learning tasks

    Argument:
    path    --  Input file path (string).

    Return:
    A list of graph_esn.Graph objects.

    """
    global currline, deb
    deb = debug
    # reset some data
    currline = 0
    # read file 
    fo = open(path, 'r')
    dataset = []

    g = __parse_graph(fo)
    while g is not None:
        dataset.append(g)
        g = __parse_graph(fo)
    fo.close()
    __deb("%i graphs read." % len(dataset))
    return dataset

def __parse_graph(fo):
    global currline 
    graph_attr = {}
    # first fixed three lines        
    molname = __next_tokens(fo, False)
    creator = __next_tokens(fo, False)
    comment = __next_tokens(fo, False)
    if None in (molname, creator, comment): # end of file reached
        return None
    # set as graph attributes
    if not molname[0] == '':
        graph_attr['sdf_molecule_name'] = " ".join(molname)
    if not creator[0] == '':
        graph_attr['sdf_creator'] = " ".join(creator)       
    if not comment[0] == '':
        graph_attr['sdf_comment'] = " ".join(comment)
    # info line
    info = __next_tokens(fo)
    __deb("(%i) [INFO] %s" % (currline, " ".join(info)))
    # check version
    version = info[-1]
    if not version == "V2000" :
        raise SDFParserError("(%i) Invalid SDF version: %s. Version \"V2000\" needed." % (currline, version))
    (anum, bnum, opt) = map(int, (info[0],info[1],info[-2])) # number of: atoms, bonds, additional info
    graph = Graph(directed=False, attr=graph_attr) # new empty Graph
    # atoms
    for _ in xrange(anum):
        t = __next_tokens(fo)
        __deb("(%i) [ATOM] %s" % (currline, " ".join(t)))
        (x,y,z) = t[0:3] # coordinates
        symbol = t[3] # chemical symbol
        v = Vertex(attr={"coord":(x,y,z), "symbol":symbol}) # set attributes 'coord' and 'symbol'
        graph.add_vertex(v)
    # bonds
    for _ in xrange(bnum): 
        t = __next_tokens(fo)
        __deb("(%i) [BOND] %s" % (currline, " ".join(t)))
        (vidx1, vidx2, btype) = map(int, t[0:3]) # from, to, type (ignore other stuff)
        # TODO: btype?
        graph.add_edge(vidx1 - 1, vidx2 -  1) # in sdf indices start from 1 -.-
    # additional info (don't trust the 'opt' data!)
    t = __next_tokens(fo)
    while not " ".join(t) == "M END":
        __deb("(%i) [OPT] %s" % (currline, " ".join(t)))
        if t[0] == "M" : # assuming format: "M <name> <N> <idx1> <val1> ... <idxN> <valN>"
            for (i,v) in zip(map(int, t[3::2]), map(float, t[4::2])):
                graph.vertices[i-1].attr[t[1]] = v # set new attribute as <name> (n.b. sdf index starts from 1)
        else: # error
            raise SDFParserError("(%i) Unrecognized extra-information line: %s" % (currline, " ".join(t)))
        t = __next_tokens(fo)
    # end of extra-info
    __deb("(%i) [END] %s" % (currline, " ".join(t)))
    # associated data items
    t = __next_tokens(fo)
    __deb("(%i) [DATA] %s" % (currline, " ".join(t)))
    while not " ".join(t) == '$$$$' :
        if t[0] == ">" and len(t) > 1:
            attrname = re.search("<(.*)>", " ".join(t[1:])).group(1) # remove brakets
            t = __next_tokens(fo)
            __deb("(%i) [ATTR] %s : %s" % (currline, attrname, " ".join(t)))
            # XXX This might be the target value!
            # N.B Parser only save it as string: XXX postprocessing is needed!
            graph.attr[attrname] = t[0] # add attribute. 
        else: # error
            raise SDFParserError("(%i) Unrecognized associated-data line: %s" % (currline, " ".join(t)))
        t = __next_tokens(fo)        
    # end of graph definition
    return graph

def __next_tokens(fo, skip_blank=True):
    global currline
    while True :
        line = fo.readline();
        currline += 1
        if not "\n" in line : # end of file
            return None
        else:
            line = line.strip()
            if len(line)>0 and line[0] == '#': # skip comment
                pass
            elif line == '':
                if not skip_blank :
                    return ['']
                else:
                    pass # skip blank line
            else: # regular line
                return re.split(r"\s+", line) # remove tabs and spaces

def __deb(txt):
    global deb
    if deb : print txt
