import re
from os.path import isfile
from g_esn.graph import Graph, Vertex

DEB = False
atom_re = re.compile("""atm\("""
    """(?P<mol>.*),"""
    """(?P<atom>.*),"""
    """(?P<sym>.*),"""
    """(?P<type>.*),"""
    """(?P<nrg>.*)"""
    """\)\.""")

bond_re = re.compile("""bond\("""
    """(?P<mol>.*),"""
    """(?P<a1>.*),"""
    """(?P<a2>.*),"""
    """(?P<type>.*)"""
    """\)\.""")
opt_re = re.compile("""(?P<prop>.*)\("""
    """(?P<mol>.*),"""
    """(?P<val>.*)"""
    """\).""")

class PlParserError(Exception):
    def __init__(self, value): self.value = value
    def __str__(self): return repr(self.value)

def parse(path, *opt):
    # check parameters
    for p in [path] + list(opt):
        if not isfile(p):
            raise PlParserError("Unexisting input file: %s" % p)
    # let's start parsing
    molname_map = {} # {molecule name : graph index}
    atomname_map = {} # {(molecule name, atom name) : vertex index}
    dataset = []
    currline = 0
    __deb('Parsing main file (atom-bond): %s' % path)
    for line in open(path):
        currline += 1
        line = line.strip()
        if line == '':
            continue # skip blank line
        # read data
        if atom_re.search(line): # atom
            m = atom_re.search(line)
            mname = m.group('mol')          # molecule name
            aname = m.group('atom')         # atom name
            sym = m.group('sym')            # atomic symbol
            atype = float(m.group('type'))  # atomic type
            nrg = float(m.group('nrg'))     # energy
            # get referred graph
            if not mname in molname_map: # create new Graph            
                g = Graph(directed=False, attr={'name':mname})
                molname_map[mname] = len(dataset) # save index
                dataset.append(g)
                __deb("(%i) New Graph: %s" % (currline, mname))
            else: # modify existing graph
                g = dataset[molname_map[mname]]
            # add new vertex
            vattr = {'name':aname, 'symbol':sym, 'type':atype, 'energy':nrg}
            v = Vertex(vattr)
            vidx = g.add_vertex(v)
            atomname_map[(mname, aname)] = vidx
            __deb("(%i) New Vertex: %s (graph: %s)" % (currline, aname, mname))
        elif bond_re.search(line): # bond
            m = bond_re.search(line)
            mname = m.group('mol')          # molecule name
            a1name = m.group('a1')          # atom name
            a2name = m.group('a2')          # atomic symbol
            btype = float(m.group('type'))  # bond type
            # check if referred molecule and vertices exist
            if not mname in molname_map:
                raise PlParserError("""(%i) Referring unexisting molecule: """
                    """%s.""" % (currline, mname))
            if not (mname, a1name) in atomname_map:
                raise PlParserError("""(%i) Referring unexisting vertex: """
                    """%s (in molecule %s)""" % (currline, a1name, mname))
            if not (mname, a2name) in atomname_map:
                raise PlParserError("""(%i) Referring unexisting vertex: """
                    """%s (in molecule %s)""" % (currline, a2name, mname))
            # create edge
            g = dataset[molname_map[mname]]
            v1idx = atomname_map[(mname, a1name)]
            v2idx = atomname_map[(mname, a2name)]
            g.add_edge(v1idx, v2idx)
            __deb("(%i) New Edge: %s - %s (graph: %s)" 
                % (currline, a1name, a2name, mname))
        else:
            raise PlParserError("(%i) Unrecognized atom/bond data." % currline)
    # additional data
    for optpath in opt:
        __deb('Parsing optional data: %s' % optpath)
        currline = 0
        for line in open(optpath):
            currline += 1
            line = line.strip()
            if line == '':
                continue # skip blank line
            # read data
            if opt_re.search(line): # atom
                m = opt_re.search(line)
                prop = m.group('prop')      # property name
                mname = m.group('mol')      # molecule name
                val = float(m.group('val')) # property value
                # check if molecule exists
                if not mname in molname_map:                
                    raise PlParserError("""(%i) Unexisting  molecule: %s\n"""
                        """File: %s""" % (currline, mname, optpath))
                # add attribute
                g = dataset[molname_map[mname]]
                g.attr[prop] = val
                __deb("(%i) New Property: %s[%s] = %f" 
                    % (currline, mname, prop, val))
            else:
                raise PlParserError("""(%i) Unrecognized optional data. """
                    """File: %s""" % (currline, optpath))
    # return the dataset
    return dataset


def __deb(txt):
    global DEB
    if DEB : print txt
