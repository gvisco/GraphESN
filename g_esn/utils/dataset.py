"""Provide functions to manipulate and getinformations about datasets (lists 
of g_esn.Graph objects).

"""
from g_esn import graph

def filter_dataset(dataset, attnames): 
    """Return new dataset obtained by removing from the dataset
    all graphs missing each attribute from given attribute list.

    Arguments:
    dataset --  Dataset to filter (list of g_esn.graph.Graph).
    attname --  List of (graph) attributes labels (list of keywords).

    Return:
    Filtered dataset (list of g_esn.graph.Graph).

    """
    return filter(lambda g: attnames in g.attr, dataset)

def purge_dataset(dataset, tasks):
    """Remove unusefull data from the dataset, e.g unlabelled data with respect 
        to given list of tasks.
    
    Arguments:
    dataset --  Dataset to purge (list of Graph).
    tasks   --  List of labels identifying a target for a task.

    """
    for g in dataset:
        purge = True
        for t in tasks:
            if t in g.attr:
                purge = False
                break
        if purge:
            dataset.remove(g)

def get_attribute_values(dataset, attribute, vertex=False):
    """Scan the dataset and return the set of values taken by given attribute.
    Returned set is an ordered list.
    Attribute can belong to the graph or to each vertex (see 'vertex' argument).

    All the graphs (or vertices if 'vertex==True') are assumed to have an
    attribute with given label.

    Arguments:
    dataset     --  List of graphs (list of g_esn.graph.Graph).
    attribute   --  Label of the attribute to check.
    vertex      --  If True then vertex attributes will be scanned, otherwise
                    graph attributes (boolean).

    Return:
    A list of sorted values (type depends of attribute's content).

    """
    if not vertex:
        return sorted(set([g.attr[attribute] for g in dataset]))
    else:
        return sorted(set().union([v.attr[attribute] for g in dataset for v in g.vertices]))

def attributes_filter(dataset, attributes):
    """Filter a dataset by removing all graphs for wich all given attributes
    are missing.

    Arguments:
    dataset     --  Dataset to filter (list of g_esn.graph.Graph).
    attributes  --  Attributes to check (list of object used as keywords).

    Return:
    Filtered dataset (list of g_esn.graph.Graph).

    """
    f = lambda g : len(set(g.attr.keys()).intersection(set(attributes))) > 0
    return filter(f, dataset)

def file_filter(dataset, path, startindex=0):
    """Return a subset of the dataset by selecting only the indices
    read by input file (one index per row).
    
    Arguments:
    dataset     --  Dataset to filter (list of Graphs).
    path        --  Path of the file containing the indices.
    startindex  --  Index numbering in the file, e.g. 1 to indicate the first 
                    graph.

    Return:
    Filtered dataset (list of Graph objects).

    """
    filtered = []
    for idx in [int(line.strip()) for line in open(path)]:
        g = dataset[idx - startindex]
        #g.attr["original_index"] = idx - startindex #DEBUG
        filtered.append(g)
    return filtered

def count_by_attribute(dataset, label, value):
    """ Count how many pattern in the dataset have a specific value for given 
    attribute (e.g. the target).

    Arguments:
    dataset --  Dataset to check (list of Graph objects).
    label   --  Attribute label to check (string).
    value   --  Value to be counted (object).

    Return:
    The number of pattern with given value for the specified attribute (int).

    """
    check = lambda g: 1 if g.attr[label] == value else 0
    return sum( map(check, dataset) )

def max_degree(dataset):
    """Return maximum in-degree and out-degree in the dataset.
    Note that for undirected graphs in-degree = out-degree.

    Arguments:
    dataset --  Input dataset (list of g_esn_graph_Graph).

    Return:
    A couple (max-in-degree, max-out-degree) both integers.
    
    """
    maxin = maxout = 0
    for g in dataset:
        for v in g.vertices:
            maxin = max(maxin, len(v.in_conn))
            maxout = max(maxout, len(v.out_conn))
    return (maxin, maxout)


def get_by_degree(dataset, deg, degtype='outdeg'):
    """Return indices of graphs in the dataset having at least one vertex with
    given degree.
    
    Arguments:
    dataset --  Input dataset (list of Graph)
    deg     --  Degree to search (integer)
    degtype --  Degree type: 'outdeg' or 'indeg' for searching out-degree or 
                in-degree respectively. For undirected graphs both have the 
                same result. (string)

    Return:
    A list of integers corresponding to indices in the dataset.

    """
    l = []
    for idx, g in enumerate(dataset):
        for v in g.vertices:
            if degtype == 'outdeg' : conn = v.out_conn
            elif degtype == 'indeg' : conn = v.in_conn
            else: raise Exception("Invalid degree type: %s" % repr(degtype))
            if len(conn) == deg :
                l.append(idx)
                break
    return l
