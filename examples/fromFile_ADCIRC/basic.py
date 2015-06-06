
# Copyright (C) 2013-2015 Lindley Graham 

"""
This module contains a set of simple classes for use by

* :py:mod:`polyadcirc.pyADCIRC.fort14_management`
* :py:mod:`polyadcirc.pyADCIRC.fort15_management`
* :py:class:`polyadcirc.run_framework.domain`
* :py:mod:`polyadcirc.run_framework.random_manningsn`
* :py:mod:`polyadcirc.pyGriddata.gridObject`
"""

class pickleable(object):
    """
    Class that makes objects easily pickleable via a dict
    """
    def __init__(self):
        super(pickleable, self).__init__()

    def __getstate__(self):
        """
        :rtype: dict
        :returns: ``self.__dict__.copy()``
        """
        odict = self.__dict__.copy()
        return odict

    def __setstate__(self, dict):
        """
        :param dict: dict to update

        ``self.__dict__.update(dict)``
        """
        self.__dict__.update(dict)

class location(pickleable):
    """
    Store x, y coordinates
    """
    def __init__(self, x, y):
        """
        Initalization
        """
        #: float, x - coodinate
        self.x = x

        #: float, y - coordinate
        self.y = y

        super(location, self).__init__()

class node(location):
    """
    Stores data specific to a single node
    """
    def __init__(self, x, y, bathymetry):
        """
        Initalization
        """
        self.bathymetry = bathymetry #: float, bathymetry at x, y
        super(node, self).__init__(x, y)

class time(pickleable):
    """
    Stores time data specific to ADCIRC model runs from the users and fort.15
    input file
    """
    def __init__(self, dt, statim, rnday, dramp):
        """
        Initalization
        """
        self.dt = dt #: float, time step (seconds)
        self.rnday = rnday #: float, total length of simulation (days)
        self.statim = statim #: float, starting time (days)
        self.dramp = dramp #: float, time constant for ramp function
        super(time, self).__init__()

        
