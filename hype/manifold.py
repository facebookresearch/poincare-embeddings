from __future__ import absolute_import, division, print_function, unicode_literals
from abc import abstractmethod


class Manifold(object):
    def __init__(self, *args, **kwargs):
        pass

    def init_weights(self, w, scale=1e-4):
        w.data.uniform_(-scale, scale)

    @staticmethod
    def dim(dim):
        return dim

    def normalize(self, u):
        return u

    @abstractmethod
    def distance(self, u, v):
        """
        Distance function
        """
        raise NotImplementedError

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport
        """
        raise NotImplementedError
