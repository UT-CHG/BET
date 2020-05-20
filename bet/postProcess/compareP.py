# Copyright (C) 2014-2020 The BET Development Team
import numpy as np
import bet.sample as samp
import bet.sampling.basicSampling as bsam
import scipy.spatial.distance as ds


class compare:
    """
    This class allows for the statistical distance between probability measures
    to be calculated.
    The probability measures may be defined by Voronoi tesselations, weighted Kernel Density Estimates,
    Gaussian Mixture Models, random variables with known parameters, and multi-dimensional
    normal distributions.
    This object can be thought of as a more flexible version of an abstraction
    of a metric, a measure of distance between two probability measures.
    It ``d(x,y)`` takes two arguments, one to the left (``x``),
    and one to the right (``y``). However, we do not enforce the properties
    that define a formal metric, instead we use the language of statistical distance.
    """
    def __init__(self, set1, set2, inputs=True, set1_init=False, set2_init=False):
        """

        Initialize comparison object.

        :param set1: Object containing left probability measure.
        :type set1: :class:`bet.sample.sample_set` or `bet.sample.discretization`lass:
        :param set2: Object containing left probability measure.
        :type set1: :class:`bet.sample.sample_set` or class:`bet.sample.discretization`
        :param inputs: If set1 and set2 are discretizations, use input sets if True and output if False.
            True by default.
        :type inputs: bool
        :param set1_init: Use initial probability measure for set1 if True. False by default.
        :type set1_init: bool
        :param set2_init: Use initial probability measure for set2 if True. False by default.
        :type set2_init: bool
        """
        self.pdfs1 = None
        self.pdfs2 = None
        self.compare_vals = None
        self.set1_init = set1_init
        self.set2_init = set2_init
        self.pdfs_zero = None

        # Extract sample sets
        if isinstance(set1, samp.discretization):
            if inputs:
                set1 = set1.get_input_sample_set()
            else:
                set1 = set1.get_output_sample_set()

        if isinstance(set2, samp.discretization):
            if inputs:
                set2 = set2.get_input_sample_set()
            else:
                set2 = set2.get_output_sample_set()

        if isinstance(set1, samp.sample_set) and isinstance(set2, samp.sample_set):
            self.set1 = set1
            self.set2 = set2
        else:
            raise samp.wrong_input("Inputs are not of valid form.")

        # Check dimensions
        if self.set1.get_dim() != self.set2.get_dim():
            raise samp.dim_not_matching("The sets do not have the same dimension.")

    def set_compare_set(self, compare_set=10000, compare_factor=0.0):
        """
        Set values where the left and right probability measures should be compared.
        If `compare_set` is of type :class:`bet.sample.sample_set`, then the values from
        that object are used. If `compare_set` is of type :class:`numpy.ndarray`, then the
        values in that array are used. If `compare_set` is of type int, then that number
        of uniformly distributed are sampled from a domain containing all of the values
        for set1 and set2. If compare_factor is set to be greater than 0, then this domain
        is increased by that proportion in every direction. Increasing the size of the
        sampling domain may catch areas of nonzero probability.

        :param compare_set: Set containing values on which to compare.
        :type compare_set: :class:`bet.sample.sample_set`, :class:`numpy.ndarray`, or int 10000 by default.
        :param compare_factor: Proportion to increase domain for sampling. Only used if `compare_set` is type int.
            0 by default.
        :type compare_factor: float
        """
        # Extract values to evaluate the probability measures.
        if isinstance(compare_set, samp.sample_set):
            if compare_set.get_dim() == self.set1.get_dim():
                compare_set.local_to_global()
                self.compare_vals = compare_set.get_values()
            else:
                raise samp.dim_not_matching("The sets do not have the same dimension.")
        elif isinstance(compare_set, np.ndarray):
            if compare_set.shape[1] == self.set1.get_dim():
                self.compare_vals = compare_set
            else:
                raise samp.dim_not_matching("The sets do not have the same dimension.")
        elif isinstance(compare_set, int):
            # Find bounds
            combined = np.vstack((self.set1.get_values(), self.set2.get_values()))
            mins = np.min(combined, axis=0)
            maxes = np.max(combined, axis=0)

            # Perform uniform random sampling.
            rv = []
            for i in range(self.set1.get_dim()):
                rv_loc = ['uniform', {}]
                delt = compare_factor * (maxes[i] - mins[i])
                rv_loc[1]['loc'] = mins[i] - delt
                rv_loc[1]['scale'] = maxes[i] - mins[i] + delt
            unif_set = bsam.random_sample_set(rv=rv_loc, input_obj=self.set1.get_dim(),
                                              num_samples=compare_set)
            self.compare_vals = unif_set.get_values()
        else:
            raise samp.wrong_input("Inputs are not of valid form.")

    def evaluate_pdfs(self):
        """
        Evaluate probability density functions associated with the probability measures at
        the comparison points.
        """
        if self.set1_init:
            self.pdfs1 = self.set1.pdf_init(self.compare_vals)
        else:
            self.pdfs1 = self.set1.pdf(self.compare_vals)

        if self.set2_init:
            self.pdfs2 = self.set2.pdf_init(self.compare_vals)
        else:
            self.pdfs2 = self.set2.pdf(self.compare_vals)

        sup1 = np.equal(self.pdfs1, 0.0)
        sup2 = np.equal(self.pdfs2, 0.0)
        self.pdfs_zero = np.sum(np.logical_and(sup1, sup2))

    def distance(self, functional='tv', normalize=False, **kwargs):
        """
        Compute the statistical distance between the probability measures
        evaluated at the comparison points.

        :param functional: functional defining type of statistical distance
        :type functional: str or a function that takes in two lists/arrays and returns
            a scalar value (measure of similarity). Accepted strings are 'tv' (total variation) the
            default, 'kl' (Kullback-Leibler),
            'mink' (minkowski), '2' (Euclidean norm), and 'hell' (Hellinger distance).
        :param normalize: whether or not to normalize the distance
        :type normalize: bool
        :param kwargs: Keyword arguments for `functional`.

        :rtype: float
        :returns: The statistical distance

        """
        # Check inputs
        if self.compare_vals is None:
            raise samp.wrong_input("Compare set needed.")
        if self.pdfs1 is None or self.pdfs2 is None:
            self.evaluate_pdfs()
        if normalize:
            self.pdfs1 = self.pdfs1 / np.sum(self.pdfs1)
            self.pdfs2 = self.pdfs2 / np.sum(self.pdfs2)
            factor = 1.0
        else:
            factor = 1.0 / (self.pdfs1.shape[0])


        if functional in ['tv', 'totvar',
                          'total variation', 'total-variation', '1']:
            dist = factor * ds.minkowski(self.pdfs1, self.pdfs2, 1, w=0.5, **kwargs)
        elif functional in ['mink', 'minkowski']:
            dist = factor * ds.minkowski(self.pdfs1, self.pdfs2, **kwargs)
        elif functional in ['norm']:
            dist = factor * ds.norm(self.pdfs1 - self.pdfs2, **kwargs)
        elif functional in ['euclidean', '2-norm', '2']:
            dist = (factor ** 0.5) * ds.minkowski(self.pdfs1, self.pdfs2, 2, **kwargs)
        elif functional in ['sqhell', 'sqhellinger']:
            dist = factor * ds.sqeuclidean(np.sqrt(self.pdfs1), np.sqrt(self.pdfs2)) / 2.0
        elif functional in ['hell', 'hellinger']:
            return np.sqrt(self.distance('sqhell'))
        elif functional in ['kl', 'k-l', 'kullback-leibler', 'entropy']:
            from scipy.stats import entropy as kl_div
            dist = kl_div(self.pdfs1, self.pdfs2, **kwargs)
        else:
            dist = functional(self.pdfs1, self.pdfs2, **kwargs)
        return dist

    def distance_marginal(self, i, interval=None, num_points=1000, compare_factor=0.0, normalize=False,
                          functional='tv', **kwargs):
        """
        Compute the statistical distance between the marginals of the probability measures
        evaluated at equally spaced points on an interval. If the interval is not defined,
        one is computed by the maximum and minimum values. This domain is extended by the proportion
        set by `compare_factor`.

        :param i: index of the marginal
        :type i: int
        :param interval: interval over which to integrate. None by default.
        :type interval: list, tuple, or :class:`numpy.ndarray`
        :param num_points: number of evaluation points. 1000 by default.
        :type num_points: int
        :param compare_factor: Proportion to increase domain. Only used if
            `interval` is None. 0 by default.
        :type compare_factor: float
        :param normalize: whether or not to normalize the probabilities to sum to 1
        :type normalize: bool
        :param functional: functional defining type of statistical distance
        :type functional: str or a function that takes in two lists/arrays and returns
            a scalar value (measure of similarity). Accepted strings are 'tv' (total variation), 'kl' (Kullback-Leibler)
            'mink' (minkowski), '2' (Euclidean norm), and 'hell' (Hellinger distance).
        :param kwargs: Keyword arguments for `functional`.

        :rtype: float
        :returns: The statistical distance

        """
        x = None
        if interval is None:
            if self.set1.get_domain() is not None and self.set2.get_domain() is not None:
                min1 = min(self.set1.get_domain()[i, 0], self.set1.get_domain()[i, 0])
                max1 = min(self.set1.get_domain()[i, 1], self.set1.get_domain()[i, 1])
                if min1 != -np.inf and max1 != np.inf:
                    delt = compare_factor * (max1 - min1)
                    x = np.linspace(min1-delt, max1+delt, num_points)
            if x is None:
                combined = np.vstack((self.set1.get_values()[:, i], self.set2.get_values()[:, i]))
                min1 = np.min(combined)
                max1 = np.max(combined)
                delt = compare_factor * (max1 - min1)
                x = np.linspace(min1 - delt, max1 + delt, num_points)
        else:
            x = np.linspace(interval[0], interval[1], num_points)

        if self.set1_init:
            pdfs1 = self.set1.marginal_pdf_init(x, i)
        else:
            pdfs1 = self.set1.marginal_pdf(x, i)

        if self.set2_init:
            pdfs2 = self.set2.marginal_pdf_init(x, i)
        else:
            pdfs2 = self.set2.marginal_pdf(x, i)

        if normalize:
            pdfs1 = pdfs1 / np.sum(pdfs1)
            pdfs2 = pdfs2 / np.sum(pdfs2)
            factor = 1.0
        else:
            factor = 1.0 / (pdfs1.shape[0]) * (x[-1] - x[0])

        if functional in ['tv', 'totvar',
                          'total variation', 'total-variation', '1']:
            dist = factor * ds.minkowski(pdfs1, pdfs2, 1, w=0.5, **kwargs)
        elif functional in ['mink', 'minkowski']:
            dist = ds.minkowski(pdfs1, pdfs2, **kwargs)
        elif functional in ['norm']:
            dist = ds.norm(pdfs1 - pdfs2, **kwargs)
        elif functional in ['euclidean', '2-norm', '2']:
            dist = (factor ** 0.5) * ds.minkowski(pdfs1, pdfs2, 2, **kwargs)
        elif functional in ['sqhell', 'sqhellinger']:
            dist = 0.5 * factor * (ds.minkowski(np.sqrt(pdfs1), np.sqrt(pdfs2), 2, **kwargs) ** 2.0)
        elif functional in ['hell', 'hellinger']:
            dist = (0.5 * factor * (ds.minkowski(np.sqrt(pdfs1), np.sqrt(pdfs2), 2, **kwargs) ** 2.0)) ** 0.5
        elif functional in ['kl', 'k-l', 'kullback-leibler', 'entropy']:
            from scipy.stats import entropy as kl_div
            dist = kl_div(pdfs1, pdfs2, **kwargs)
        else:
            dist = functional(pdfs1, pdfs2, **kwargs)

        return dist

    def distance_marginal_quad(self, i, interval=None, compare_factor=0.0,
                               functional='tv', **kwargs):
        """
        Compute the  statistical distance between the marginals of the probability measures
        by integrating using `scipy.integrate.quadrature`.. If the interval is not defined,
        one is computed by the maximum and minimum values. This domain is extended by the proportion
        set by `compare_factor`.

        :param i: index of the marginal
        :type i: int
        :param interval: interval over which to integrate. None by default.
        :type interval: list, tuple, or :class:`numpy.ndarray`
        :param compare_factor: Proportion to increase domain. Only used if
            `interval` is None. 0 by default.
        :type compare_factor: float
        :param functional: functional defining type of statistical distance
        :type functional: str or a function that takes in two lists/arrays and returns
            a scalar value (measure of similarity). Accepted strings are 'tv' (total variation),
            'mink' (minkowski), '2' (Euclidean norm), 'kl' (Kullback-Leibler) and 'hell' (Hellinger distance).
        :param kwargs: Keyword arguments for `scipy.integrate.quadrature`.

        :rtype: float
        :returns: The statistical distance

        """
        from scipy.integrate import quadrature
        if interval is None:
            if self.set1.get_domain() is not None and self.set2.get_domain() is not None:
                min1 = min(self.set1.get_domain()[i, 0], self.set1.get_domain()[i, 0])
                max1 = min(self.set1.get_domain()[i, 1], self.set1.get_domain()[i, 1])
                if min1 != -np.inf and max1 != np.inf:
                    delt = compare_factor * (max1 - min1)
                    interval = [min1-delt, max1 + delt]
            if interval is None:
                combined = np.vstack((self.set1.get_values()[:, i], self.set2.get_values()[:, i]))
                min1 = np.min(combined)
                max1 = np.max(combined)
                delt = compare_factor * (max1 - min1)
                interval = [min1 - delt, max1 + delt]
                
        if self.set1_init:
            pdf1 = self.set1.marginal_pdf_init
        else:
            pdf1 = self.set1.marginal_pdf

        if self.set2_init:
            pdf2 = self.set2.marginal_pdf_init
        else:
            pdf2 = self.set2.marginal_pdf

        if functional in ['tv', 'totvar',
                          'total variation', 'total-variation', '1']:
            def error(x):
                return np.abs(pdf1(x, i) - pdf2(x, i))
            return 0.5 * quadrature(error, interval[0], interval[1], **kwargs)[0]
        elif functional in ['euclidean', '2-norm', '2']:
            def error(x):
                return (pdf1(x, i) - pdf2(x, i))**2
            return (quadrature(error, interval[0], interval[1], **kwargs)[0])**0.5
        elif functional in ['norm']:
            def error(x):
                return pdf1(x, i) - pdf2(x, i)

            return quadrature(error, interval[0], interval[1], **kwargs)[0]
        elif functional in ['sqhell', 'sqhellinger']:
            def error(x):
                return 0.5 * (np.sqrt(pdf1(x, i)) - np.sqrt(pdf2(x, i)))**2
            return quadrature(error, interval[0], interval[1], **kwargs)[0]
        elif functional in ['hell', 'hellinger']:
            return np.sqrt(self.distance_marginal_quad(i, interval, compare_factor=0,
                                                       functional="sqhell", **kwargs))
        elif functional in ['kl', 'k-l', 'kullback-leibler', 'entropy']:
            def error(x):
                return pdf1(x, i) * np.log(pdf1(x, i)/pdf2(x, i))

            return quadrature(error, interval[0], interval[1], **kwargs)[0]
        else:
            def error(x):
                return functional(pdf1(x, i), pdf2(x, i))
            return quadrature(error, interval[0], interval[1], **kwargs)[0]
