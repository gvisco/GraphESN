#!/usr/bin/env python

import sys, tempfile, os, re, os.path, shutil
from xml.dom import minidom, Node
import igraph
from PyQt4 import QtGui, QtCore

try: 
    import IPython
except ImportError:
    I_SHELL = False
else:
    I_SHELL = True

from g_esn.graph import *
from g_esn.parsers import sdf_parser, gph_parser

TEMPFILE = tempfile.gettempdir() + os.sep + "gviewer-current-graph.svg"
MISSING_IMG = "missing.svg"
LAYOUTS = ['kk', 'fr', 'tree', 'circle', 'random']
VERTEX_COLORS = {'sel':'red', 'def':'grey'}
LABEL_COLORS = {'sel':'red', 'def':'black'}
LABEL_ATTR = 'symbol'
(INIT_W, INIT_H) = (800, 400)
IMG_SIDE = 600
IMG_MARGIN = 20
(VERTEX_MIN, VERTEX_MAX) = (1, 28)
(LABEL_MIN, LABEL_MAX) = (1, 32)
(INIT_VSIZE, INIT_LSIZE) = (50, 50) # percent

class ImageViewer(QtGui.QWidget):
    def __init__(self, parent, image=None):
        QtGui.QWidget.__init__(self, parent)
        self.img = image

    def setImage(self, img):
        self.img = img
        self.ratio = float(img.width()) / float(img.height())
        self.repaint()

    def paintEvent(self, e):
        if self.img :
            wratio = float(self.width()) / float(self.height())
            img = self.img.scaledToHeight(self.height(), 1) if wratio >= self.ratio else self.img.scaledToWidth(self.width(), 1)
            (x,y) = ((self.width() - img.width())/2, (self.height() - img.height())/2)
            qp = QtGui.QPainter()
            qp.begin(self)                        
            qp.drawImage(x, y, img)
            qp.end()


class GraphViewer(QtGui.QMainWindow):
    def __init__(self, dataset=None):
        QtGui.QWidget.__init__(self, None)
        self.dataset = dataset
        self.fname = ""
        self.xmldoc = None
        # user selection
        self.current_graph = 0
        self.current_vertex = 0
        self.current_layout = LAYOUTS[0]
        self.current_vertexsize = INIT_VSIZE
        self.current_labelsize = INIT_LSIZE
        # gui elements
        self.gcombo = None # QComboBox for graph selection
        self.vcombo = None # QComboBox for vertex selection
        self.imgview = None # ImageViewer widget
        self.graph_details = None # QTextEdit to show graph details
        self.vertex_details = None # QTextEdit to show vertex details
        # init UI
        self.__init_ui()
        # init menu
        self.__init_menu()
        # setup window
        self.resize(INIT_W, INIT_H) # widget size
        self.setWindowTitle('Graph Dataset Visualization') # window title        
        self.center() # center the screen        
        self.__change_view() # display something

    def __init_menu(self):
        # open
        openf = QtGui.QAction('Open', self)
        openf.setShortcut('Ctrl+O')
        openf.setStatusTip('Open file...')
        self.connect(openf, QtCore.SIGNAL('triggered()'), self.__open) 
        # exit
        exit = QtGui.QAction('Exit', self)
        exit.setShortcut('Ctrl+Q')
        exit.setStatusTip('Exit application')
        self.connect(exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        # setup menu
        menubar = self.menuBar()        
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(openf)
        filemenu.addAction(exit)

    def __init_ui(self):
        top = self.__init_top()
        center_l = self.__init_center_left()
        center_r = self.__init_center_right()
        # central panel
        mainsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        mainsplitter.addWidget(center_l)
        mainsplitter.addWidget(center_r)
        mainsplitter.setSizes([INIT_W * 2/3, INIT_W * 1/3])
        # main layout
        vbox = QtGui.QVBoxLayout() 
        vbox.addWidget(top)
        vbox.addWidget(mainsplitter)
        mainframe = QtGui.QFrame(self)
        mainframe.setLayout(vbox)
        self.setCentralWidget(mainframe)

    def __init_top(self):
        # graph selection
        self.gcombo = QtGui.QComboBox(self) # graph indices
        self.gcombo.setMinimumContentsLength(3)
        self.gcombo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToMinimumContentsLength)
        self.__reset_graph_combo()
        self.connect(self.gcombo, QtCore.SIGNAL('activated(QString)'), self.__graph_changed)
        # vertex selection
        self.vcombo = QtGui.QComboBox(self) # vertex indices
        self.vcombo.setMinimumContentsLength(3)
        self.vcombo.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToMinimumContentsLength)
        self.__reset_vertex_combo()
        self.connect(self.vcombo, QtCore.SIGNAL('activated(QString)'), self.__vertex_changed)
        # layout selection
        lcombo = QtGui.QComboBox(self) # layouts
        lcombo.addItems(LAYOUTS)
        self.connect(lcombo, QtCore.SIGNAL('activated(QString)'), self.__layout_changed)
        # vertex size 
        vsslider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        vsslider.setFocusPolicy(QtCore.Qt.NoFocus)
        vsslider.setValue(INIT_VSIZE)
        self.connect(vsslider, QtCore.SIGNAL('valueChanged(int)'), self.__vertexsize_changed)
        # label size 
        lsslider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        lsslider.setFocusPolicy(QtCore.Qt.NoFocus)
        lsslider.setValue(INIT_LSIZE)
        self.connect(lsslider, QtCore.SIGNAL('valueChanged(int)'), self.__labelsize_changed)
        # reload button
        relbtn = QtGui.QPushButton('Reload', self)
        self.connect(relbtn, QtCore.SIGNAL('clicked()'), self.__change_view)
        # layout setup
        hbox = QtGui.QHBoxLayout() # top layout
        wlist = [QtGui.QLabel("Graph:"), self.gcombo, 
                QtGui.QLabel("Vertex:"), self.vcombo, 
                QtGui.QLabel("Layout:"), lcombo,
                QtGui.QLabel("Vertex Size:"), vsslider,
                QtGui.QLabel("Label Size:"), lsslider]
        for widget in wlist:
            hbox.addWidget(widget)
        hbox.addStretch(1)
        hbox.addWidget(relbtn) # reload button
        # top frame 
        topframe = QtGui.QFrame(self)
        topframe.setFrameShape(QtGui.QFrame.StyledPanel)
        sp = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Maximum)
        topframe.setSizePolicy(sp)
        topframe.setLayout(hbox)        
        return topframe

    def __init_center_left(self):
        self.imgview = ImageViewer(self)
        return self.imgview

    def __init_center_right(self):
        # top: graph details
        self.graph_details = QtGui.QTextEdit(self) # graph details
        vbox1 = QtGui.QVBoxLayout()
        vbox1.addWidget(QtGui.QLabel("Graph Attributes:")) 
        vbox1.addWidget(self.graph_details)
        rt = QtGui.QFrame(self)
        rt.setLayout(vbox1)
        # bottom: vertex details
        self.vertex_details = QtGui.QTextEdit(self) # vertex details
        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(QtGui.QLabel("Vertex Attributes:")) 
        vbox2.addWidget(self.vertex_details)
        rb = QtGui.QFrame(self)
        rb.setLayout(vbox2)
        # vertical splitter
        vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        vsplitter.addWidget(rt)
        vsplitter.addWidget(rb)
        return vsplitter               

    def __reset_graph_combo(self):
        self.gcombo.clear()
        if not self.dataset is None:
            items = [str(i) for i in xrange(len(self.dataset))]
            self.gcombo.addItems(items)
            self.gcombo.adjustSize()
            self.current_graph = 0
    
    def __reset_vertex_combo(self):
        self.vcombo.clear()
        if not self.dataset is None:
            items = [str(i) for i in xrange(len(self.dataset[self.current_graph].vertices))]
            self.vcombo.addItems(items)
            self.current_vertex = 0

    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
    
#------------------------- Events

    def __open(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, caption='Open file', 
                filter="Molecule files (*.sdf *.gph);;All (*.*)", 
                selectedFilter="Molecule files (*.sdf *.gph);;All (*.*)")
        if not fname == QtCore.QString(''):
            try:
                ext = os.path.splitext(str(fname))[1]
                if   ext in ('.sdf', '.sd') : 
                    self.dataset = sdf_parser.parse(fname)
                elif ext == '.gph' : 
                    self.dataset = gph_parser.parse(fname)
                else : raise Exception('Invalid file extension')
            except Exception as e:
                print "Exception: %s" % str(e)
                QtGui.QMessageBox.information(self,"Something goes wrong", 
                        "Invalid dataset. \n\nPlease verify or choose another file." )
            else:
                self.filename = fname
                self.__reset_graph_combo()
                self.__graph_changed('0')

    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, "Don't be nervous", "Are you sure to quit?", 
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            if os.path.exists(TEMPFILE): 
                os.remove(TEMPFILE)
            event.accept()
        else:
            event.ignore()

    def __vertexsize_changed(self, value):
        #oldsize = self.__get_vertex_size()        
        self.current_vertexsize = value
        if not self.dataset is None:
            # modify svg file
            newsize = self.__get_vertex_size()
            root = self.xmldoc.documentElement
            gs = root.getElementsByTagName('g')
            for g in gs:
                if g.getAttribute('id') == 'vertices':
                    vertices = g.getElementsByTagName('g')
                    for v in vertices:
                        v.getElementsByTagName('circle')[0].setAttribute('r', str(newsize))
             # write file
            f = open(TEMPFILE, "w")
            self.xmldoc.writexml(f)
            f.close()
            self.__update_view()

    def __labelsize_changed(self, value):
        # save current value
        self.current_labelsize = value
        if not self.dataset is None:
            newsize = self.__get_label_size() 
            root = self.xmldoc.documentElement
            style = root.getElementsByTagName('defs')[0].getElementsByTagName('style')[0]
            olddata = style.childNodes[1].wholeText
            # replace font size
            expr = re.compile(r"(?P<start>.*#vertices text { .* font-size: )(?P<repl>\d*)(?P<end>px; .*}.*)", re.S | re.M)
            m = expr.match(olddata)
            newdata = m.group('start') + str(newsize) + m.group('end')
            # update dom
            cdatasec = self.xmldoc.createCDATASection(newdata)
            style.replaceChild(cdatasec, style.childNodes[1])
            # write file
            f = open(TEMPFILE, "w")
            self.xmldoc.writexml(f)
            f.close()
            self.__update_view()

    def __vertex_changed(self, text):
        oldsel = self.current_vertex
        self.current_vertex = int(text)
        # modify highlighted vertex
        root = self.xmldoc.documentElement
        gs = root.getElementsByTagName('g')
        for g in gs:
            if g.getAttribute('id') == 'vertices':
                vs = g.getElementsByTagName('g')
                vs[oldsel].getElementsByTagName('circle')[0].setAttribute('fill', 'grey')
                vs[self.current_vertex].getElementsByTagName('circle')[0].setAttribute('fill', 'red')
        f = open(TEMPFILE, "w")
        self.xmldoc.writexml(f)
        f.close()
        self.__update_view()

    def __graph_changed(self, text):
        self.current_graph = int(text)
        self.__reset_vertex_combo()
        self.__change_view()
    
    def __layout_changed(self, text):
        self.current_layout = str(text)
        self.__change_view()

#------------------------- Update View

    def __get_label_size(self):
        return int(self.current_labelsize / 100.0 * (LABEL_MAX - LABEL_MIN) + LABEL_MIN)

    def __get_vertex_size(self):
        return int(self.current_vertexsize / 100.0 * (VERTEX_MAX - VERTEX_MIN) + VERTEX_MIN)

    def set_dataset(self, datset):
        self.dataset = dataset
        self.fname = ""
        self.__reset_graph_combo()
        self.__graph_changed('0')

    def __update_view(self):
        if not self.dataset is None:
            img = QtGui.QImage(TEMPFILE)        
            self.imgview.setImage(img)
            # details
            g = self.dataset[self.current_graph]
            v = g.vertices[self.current_vertex]
            self.graph_details.setText(repr(g.attr))
            self.vertex_details.setText(repr(v.attr))
    
    def __change_view(self):
        if not self.dataset is None:
            # save new image
            labelsize = self.__get_label_size()
            vertexsize = self.__get_vertex_size()
            try:
                save_image(self.dataset[self.current_graph], TEMPFILE, 
                        LABEL_ATTR, self.current_layout, 
                        self.current_vertex, 0, labelsize, margin=IMG_MARGIN)            
                # XXX dirty hack to avoid edges to appear 'broken' when vertices are scaled
                self.xmldoc = minidom.parse(TEMPFILE)
                root = self.xmldoc.documentElement
                gs = root.getElementsByTagName('g')
                for g in gs:
                    if g.getAttribute('id') == 'vertices':
                        vertices = g.getElementsByTagName('g')
                        for v in vertices :
                            v.getElementsByTagName('circle')[0].setAttribute('r', str(vertexsize))
                # add a margin (considering vertex expansion)
                (w, h) = map(int, (root.getAttribute('width'), root.getAttribute('height')))
                (neww, newh) = (w + 2 * VERTEX_MAX, h + 2 * VERTEX_MAX)
                root.setAttribute('width', str(neww))
                root.setAttribute('height', str(newh))
                # center the graph (with respect to new size)
                g = filter(lambda n: n.nodeType == n.ELEMENT_NODE and n.hasAttribute('transform'), root.childNodes)[0]
                g.setAttribute('transform', 'translate(%f,%f)' % (neww/2.0, newh/2.0))
                # write file
                f = open(TEMPFILE, "w")
                self.xmldoc.writexml(f)
                f.close()                
            except: # XXX needed to avoid a bug in igraph (it doesn't show graphs with only one node)
                missing_path = os.path.dirname( __file__ ) + os.sep + MISSING_IMG
                shutil.copyfile(missing_path, TEMPFILE)
            self.__update_view()

            
def as_IGraph(g, label):
    ig = igraph.Graph(n=len(g.vertices), directed=False)
    # edges
    edges = [(i,j) for (i,v) in enumerate(g.vertices) for j in v.out_conn]
    #edges = [(i,j) for i in xrange(len(g.vertices)) for j in g.vertices[i].out_conn]
    ig.add_edges(edges)
    # names
    names = [v.attr[label] for v in g.vertices]
    ig.vs["label"] = names
    return ig

def save_image(g, path, label, layout="kk", selected=-1, vertexsize=14, labelsize=18, *args, **kwds):
    ig = as_IGraph(g, label)
    ig.vs['color'] = [VERTEX_COLORS['sel'] if idx==selected else VERTEX_COLORS['def'] for idx in xrange(len(g.vertices))]
    ig.vs['label_color'] = [LABEL_COLORS['sel'] if idx==selected else LABEL_COLORS['def'] for idx in xrange(len(g.vertices))]
    #ig.vs['label_size'] = [labelsize] * len(g.vertices)
#    l = ig.layout(layout)
    l = ig.layout('fr')
    #igraph.plot(ig, path, layout=l, *args, **kwds)
    ig.write_svg(path, layout=l)#, vertex_size=vertexsize, font_size=labelsize)
    
        

if __name__=="__main__":
    dataset = None
    if len(sys.argv) > 1 :
        #dataset = sdf_parser.parse(sys.argv[1])
        dataset = gph_parser.parse(sys.argv[1])
    
    app = QtGui.QApplication(sys.argv)
    gv = GraphViewer(dataset)
    gv.show()

    # embed a IPython interactive shell
    if I_SHELL : # run IPython interactive shell
        from IPython.Shell import IPShellQt4
        sys.argv = sys.argv[0:1]
        ipshell = IPShellQt4(user_ns=dict(dataset=dataset, app=gv), argv=["-nobanner"]) # export dataset
        ipshell.mainloop()

    sys.exit(app.exec_())

