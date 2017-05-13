"""Package contains modules that ease the use, the manipulation and the analysis
of models, data, results.

It's worth noting that none of these modules is needed by the g_esn "core".

"""
# Package g_esn.utils

__all__ = [
    "data_saver",   # Facilities for storing and saving data    
    'dataset',      # Manipulation and info about datasets
    'matrices',     # Creation and manipulation of scipy.arrays
    'performance',  # Performance and data extraction (e.g. subnetwork's targets)
    'plot',         # Save plot and other files
    ]
