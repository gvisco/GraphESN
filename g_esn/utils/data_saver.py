"""Module contains the class DataSaver to ease data storage."""

import pickle
import os
import datetime

class DataSaver(object):
    """Provide facilities to store and save data.

    All methods are class methods in order to mantain a static "global"" state, 
    allowing many components of the application to save data in a single object.

    Data is saved, retrievied and modified using keywords, as in a dictionary.
    Also named global counters are provided in order to allow different 
    (subsequent) instances of a same class to use a unique keyword.
 
    """

    __data = {}
    __counters = {}
    dirpath = None # Ends with a dir separator

    @classmethod
    def init(cls, dirpath=".", newdir=True):
        """Reset/Initialize the DataSaver.
        Create the destination directory if needed and clear existing data
        and counters.

        Arguments:
        dirpath --  Destination directory path. (string)
        newdir  --  If True create new directory (into dirpath) where data will
                    be saved. (boolean)

        """
        dirpath = dirpath if dirpath[-1] == os.sep else dirpath + os.sep        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if newdir:
            now = datetime.datetime.now()
            dirpath = dirpath + now.strftime("%y-%m-%d_%H-%M-%S") + os.sep
            os.mkdir(dirpath)
        cls.dirpath = dirpath
        cls.__data.clear()
        cls.__counters.clear()

    @classmethod
    def set_field(cls, name, val):
        """Init new field."""
        cls.__data[name] = val

    @classmethod
    def get_field(cls, name):
        """Get a field."""
        return cls.__data[name]

    @classmethod
    def contains_field(cls, field_id):
        """Check if given field (keyword) exists"""
        return field_id in cls.__data        
            
    @classmethod
    def incr_counter(cls, cname):
        """Increase counter and return new value.
        Create new counter if it doesn't exists.

        """
        if not cname in cls.__counters: cls.__counters[cname] = -1
        cls.__counters[cname] += 1
        return cls.__counters[cname]

    @classmethod
    def get_counter(cls, cname):
        """Return counter value."""
        return cls.__counters[cname]

    @classmethod
    def contains_counter(cls, cname):
        """Check if given counter exists."""
        return cname in cls.__counters

    @classmethod
    def clear_data(cls):
        """Clear saved data and current counters"""
        cls.__data.clear()
        cls.__counters.clear()

    @classmethod
    def save_string(cls, name, string):
        """Save a string to file"""
        fd = open(cls.dirpath + name + '.txt', 'wb')
        fd.write(string)
        fd.close()        

    @classmethod
    def save(cls, name, string=False):
        """Save and clear specified field."""
        if not string:
            pickle.dump(cls.__data[name], open(cls.dirpath + name + '.p', 'wb'))
        else:
            fd = open(cls.dirpath + name + '.txt', 'wb')
            fd.write(str(cls.__data[name]))
            fd.close()
        del cls.__data[name]

    @classmethod
    def save_all(cls, dirpath="."):
        """Save stored data to files."""
        for n, v in cls.__data.items():
            pickle.dump(v, open(cls.dirpath + n + '.p', 'wb'))
        print "Data saved to: %s" % dirpath
