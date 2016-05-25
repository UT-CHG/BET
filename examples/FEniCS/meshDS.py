#!/usr/bin/en python

from dolfin import * 
from numpy import *


class meshDS(object):

    """Docstring for meshDS. """

    def __init__(self,mesh):
        """TODO: to be defined1.

        :mesh: reads a fenics mesh object 

        """
        self._mesh = mesh
        self.node_elem = {} # empty dictionary of node to elements connectivity
        self.edges_elem = {} # empty dictionary of edges to elements connectivity

        # initialize the mesh and read in the values
        self._mesh.init()
        self._dim = self._mesh.topology().dim()
        self.num_nodes = self._mesh.num_vertices()
        self.num_elements = self._mesh.num_cells()
        self.num_edges = self._mesh.num_edges()

    def getNodes(self):
        """TODO: Docstring for getNodes.
        :returns: num of nodes in the mesh

        """
        return self.num_nodes
    def getElements(self):
        """TODO: Docstring for getElements.
        :returns: number of elements in the mesh 

        """
        return self.num_elements

    def getEdges(self):
        """TODO: Docstring for getElements.
        :returns: number of elements in the mesh 

        """
        return self.num_edges
    def getElemToNodes(self):
        """TODO: Docstring for getElemToNodes.
        :returns: Elements - Nodes Connectivity array of array

        """
        return self._mesh.cells()
    def getNodesToElem(self):
        """TODO: Docstring for getNodesToElem.
        :returns: returns Nodes to Element connectivity as a dictionary
        where nodes_elem[i] is an array of all the elements attached to node i

        """
        for nodes in entities(self._mesh,0):
            self.node_elem[nodes.index()] = nodes.entities(self._dim)
        return self.node_elem
    def getElemVCArray(self):

        """TODO: Docstring for getElemVCArray.
        :returns: array of element volume and and an array of element centroid object 
        Thus elem_centroid_array[i][0] means the x co-ordinate of the centroid for element number i
        Thus elem_centroid_array[i][1] means the y co-ordinate of the centroid for element number i
        """

        elem_vol_array = empty((self.num_elements),dtype=float)
        elem_centroid_array = empty((self.num_elements),dtype=object)

        cell_indx = 0
        for node_list in self._mesh.cells():
            # First get the cell object corresponding to the cell_indx
            cell_obj = Cell(self._mesh,cell_indx)
            # Find the cell volume and cell centroid
            elem_vol_array[cell_indx] = cell_obj.volume()
            elem_centroid_array[cell_indx] = cell_obj.midpoint()
            # update cell index
            cell_indx = cell_indx + 1
        return elem_vol_array,elem_centroid_array

