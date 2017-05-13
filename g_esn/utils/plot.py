"""Some function to save plots, images or other files useful when describing 
the behaviour of a model or other data.

Requires: matplotlib.pyplot

"""

import os
import matplotlib
matplotlib.use('Agg')
from  matplotlib.pyplot import (figure, plot, boxplot, errorbar, scatter, 
        legend, axis, savefig, ylabel, xlabel, cm, title, suptitle)

def save_plot(data, fname="plot.pdf", axis_labels=None, show_legend=True, 
        plot_title=None, plot_suptitle="None"):
    """Plot a set of curves and save resulting image.

    Arguments:
    data            --  List of lists. Each row is a curve.
                        or
                        A list of floats in only one curve has to be displayed.
    fname           --  Name of the output file. Extension matters.
    axis_labels     --  Axes names: (x label, y label) or None
    show_legend     --  Display a legend or not (boolean).
    plot_title      --  Plot title (string).
    plot_suptitle   --  Plot suptitle (string).

    """    
    figure()
    if isinstance(data[0], list): # multiple curves
        for idx, curve in enumerate(data):
            plot(curve, label=str(idx+1))
        if show_legend: 
            legend(loc="best")
    else: # only one curve
        plot(data)        
    if not axis_labels is None:
        xlabel(axis_labels[0])
        ylabel(axis_labels[1])    
    if not plot_title is None:
        title(plot_title)        
    if not plot_suptitle is None:
        suptitle(plot_suptitle)
    savefig(fname)

def save_box_plot(data, fname="box_plot.pdf", axis_labels=None, 
    plot_title=None, plot_suptitle="None"):
    """Save a box-plot of given data.

    Box-plot display the minimum and the maximum, quartiles 1, 2 and 3 and
    percentiles 2th and 98th. Also the mean is displayed as a curve.

    Arguments:
    data            --  List of lists. Each row is a curve.
    fname           --  Name of the output file. Extension matters.
    axis_labels     --  Axes names: (x label, y label) or None
    plot_title      --  Plot title (string)
    plot_suptitle   --  Plot suptitle (string).

    """
    figure()
    transpose = lambda l: [[l[j][i] for j in range(len(l))] for i in range(len(l[0]))]
    tr_data = transpose(data)
    # boxplot
    boxplot(tr_data)
    # plot the mean as a curve    
    avg = lambda l: sum(l)/len(l) 
    avg_data = map(avg, tr_data)
    plot(avg_data)
    # axis labels
    if not axis_labels == None:
        xlabel(axis_labels[0])
        ylabel(axis_labels[1])    
    if not plot_title is None:
        title(plot_title)
    if not plot_suptitle is None:
        suptitle(plot_suptitle)
    savefig(fname)

def save_error_plot(data, fname="error_plot.pdf", axis_labels=None, 
        plot_title=None, plot_suptitle=None):
    """Save an error-plot based on given data.

    Error-plot display the average value plus/minus the standard deviation 
    for each point.

    Arguments:
    data            --  List of lists. Each row is a curve.
    fname           --  Name of the output file. Extension matters.
    axis_labels     --  Axes names: (x label, y label) or None
    plot_title      --  Plot title (string)
    plot_suptitle   --  Plot suptitle (string).

    """
    from math import sqrt
    figure()
    transpose = lambda l: [[l[j][i] for j in range(len(l))] for i in range(len(l[0]))]
    avg = lambda l: sum(l)/len(l) 
    st_dev = lambda (values, mean): sqrt( sum([(v-mean)**2 for v in values])/len(values) )
    tr_data = transpose(data)     
    avg_data = map(avg, tr_data)
    stdev_data =  map(st_dev, zip(tr_data, avg_data))
    errorbar(range(len(tr_data)), avg_data, stdev_data)   
    # axis labels
    if not axis_labels == None:
        xlabel(axis_labels[0])
        ylabel(axis_labels[1])    
    if not plot_title is None:
        title(plot_title)
    if not plot_suptitle is None:
        suptitle(plot_suptitle)
    savefig(fname)

def scatter_plot(points, values=None, discrete=False, fname="scatter_plot.pdf",
    axis_labels=None, valrange=None, show_legend=True, color_map=cm.gray, 
    plot_title=None, plot_suptitle=None):
    """ Save a scatter plot as image.

    Arguments:
    points          --  List of couples (x, y).
    values          --  A list containing values associated with each point (a 
                        class attribute or a continuous value) or None.
    discrete        --  If values is not None, tells whether the values have to
                        be considered as discrete data or not. (boolean)
    fname           --  Destination file name. Extension matters. (string)
    axis_labels     --  Axes labels. (x label, y label) or None.  
    valrange        --  Couple (minvalue, maxvalue). If None then axes are 
                        scaled to automatically fit the data.
                    Plot is always squared (i.e. both axes have the same range).
    show_legend     --  If values is not None and data is discrete, choose to 
                        display a legend or not (boolean).
    color_map       --  A Colormap to be used when values is not None and 
                        corresponding data is not discrete 
                        (matplotlib.colors.Colormap).
    plot_title      --  Plot title (string)
    plot_suptitle   --  Plot suptitle (string).

    """
    figure()
    x, y = zip(*points) # unzip points
    # axis
    maxval, minval = valrange if valrange else (max(x+y) * 1.1 , min(x+y) * 1.1)
    axis([minval, maxval, minval, maxval])
    # axis labels
    if axis_labels:
        xlabel(axis_labels[0])
        ylabel(axis_labels[1])    
    # plot
    if values:
        if discrete: # points belongs to classes
            m = ['s', 'o', '^', '>', 'v', '<', 'd', 'p', 'h', '8', '+', 'x']
            c = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'grey']
            clnames = sorted(set(values)) # class names
            for cidx, cn in enumerate(clnames):
                pts = [points[idx] for idx, el in enumerate(values) if el == cn]
                x, y = zip(*pts) # unzip
                scatter(x, y, marker=m[cidx % len(m)], c=c[cidx % len(c)], label=str(cn))
            # legend
            if show_legend: 
                legend(loc='best')
        else: # continuous values -> using a colormap
            scatter(x, y, c=values, cmap=color_map)
    else: # no values attached
        scatter(x, y)
    if not plot_title is None:
        title(plot_title)
    if not plot_suptitle is None:
        suptitle(plot_suptitle)
    # save
    savefig(fname)

def anim_scatter_plot(points_list, values, 
    fname="anim_scatter.mpg", fps=2, *args, **kwargs):
    """Generate a MPG video showing multiple scatter plots.

    Arguments:
    points_list     --  A list of lists, one for each frame/plot. Each sub-list
                        contains points to plot as pairs (x,y).
    values          --  Classes corrensponding to each point 
                        (see multiclass_scatter_plot). A list of real values.
    fname           --  Name of the MPG output file (string).
    fps             --  Frames per second (int).
    others          --  Parameters used when calling the 
                        'multiclass_scatter_plot' function.
                        Warning: do not specify any 'fname' parameter.

    Requires: 
    Command 'mencoder' has to be installed.

    See: multiclass_scatter_plot

    """
    print "Genrating temp images"
    for idx, pts in enumerate(points_list):
        print "\tPlot %i of %i" % (idx, len(points_list))
        scatter_plot(pts, values, "_tmp_%i.png" % idx, *args, **kwargs)
    print "Creating animation"  
    os.system("mencoder 'mf://_tmp_*.png' -mf type=png:fps=%i -ovc\
        lavc -lavcopts vcodec=wmv2 -oac copy -o %s" % (fps, fname))
    print "Removing temp files"
    os.system("rm -f _tmp_*.png")

def texify_table(table, labels=None, row_labels=None, align='c'):
    """Represent a table (list of lists) as TeX tabular.
    Return corresponding string.

    Arguments:
    table       --  A list of lists, i.e. the table.
    labels      --  List or tuple containing one label for each column, or None.
    row_labels  --  List or tuple containing one label for each row, or None.
    align       --  Alignment character.

    Return:
    A string in TeX syntax (needs package 'booktabs' to be rendered).

    """
    rows = len(table)
    cols = len(table[0])
    if labels is not None and len(labels) != cols:
        raise Exception("Invalid argument value: labels.")
    if row_labels is not None and len(row_labels) != rows:
        raise Exception("Invalid argument value: row_labels.")
    # begin table
    s = "\\begin{tabular}{"
    if row_labels is not None: s += 'l|'
    s += align * cols
    s += "}\n"
    s += "\\toprule\n"
    # header
    if labels is not None:
        if row_labels is not None: s += ' & '
        s += " & ".join(labels)
        s += " \\\\ \n"
        s += "\\midrule\n"
    # table
    for idx, row in enumerate(table):
        if row_labels is not None: s += row_labels[idx] + " & "
        s += " & ".join(map(str, row))
        s += " \\\\ \n"
    # end table
    s += "\\bottomrule\n"
    s += "\\end{tabular}"    
    return s
