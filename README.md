# GraphESN
The repository contains the code I wrote for my master thesis "Constructive Reservoir Computing neural models for structured domains" (available [here](https://etd.adm.unipi.it/t/etd-01232012-162826/) in italian). The thesis is about a new set of neural models that apply the reservoir computing approach to problems where the input data can be represented as a graph.

## Repository Content
- `g_esn`: The source code package implementing the GraphESN and the Constructive-GraphESN models in Python 2.6.
- `example`: A simple tutorial showing how to use the g_esn library. To run the code type `python tutorial.py`. It requires g_esn and it's dependencies to be installed and availables for the Python interpreter.

## Bonus (Data Viewer)
The directory `g_esn/viewer` contains a standalone application to visualize parsed data.

It requires: PyQt4, igraph, IPython (optional), g_esn (at least the `Graph` class and the `g_esn.parsers` package). 

To launch the application, type: `python data_viewer`.

### How to use the data_viewer
To open a dataset choose `File > Open` and select an SDF or GPH file.
In order to load a Progol dataset, ipython must be installed:
- Launch the program
- Use the binded ipython console to load the dataset (see `pl_parser` documentation).
  ```files = ['atom_bond.pl', 'logp.pl', 'lumo.pl', 'ind1.pl', 'inda.pl', 'log_mutag.pl']
     dset = pl_parser.parse(*files)
     ```
- Set the parsed data as the displayed dataset (variable: app.dataset):
  ```app.dataset = dset```
