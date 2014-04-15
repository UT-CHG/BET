"""
This subpackage provides classes and methods for calulating the probability
measure $P_{\Lambda}$.

* :mod:`~bet.calculateP.q_singleV` provides a skeleton class and calculates the
    probability for a set of emulation points.
* :mod:`~bet.calculateP.q_singleVex` calculates the exact volumes of the
    interior voronoi cells and estimates the volumes of the exterior voronoi
    cells by using a set of bounding points
* :mod:`~bet.calculateP.q_singleVmc` estimates the volumes of the voronoi cells
    using MC integration

"""
__all__ = ['q_singleV', 'q_singleVex', 'q_singleVmc']
