#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
import torch


class Manifold(object):
    """
    Base class for all manifolds.
    """
    def __init__(self, *args, **kwargs):
        pass

    def init_weights(self, w, scale=1e-4):
        """
        Initialize the weights of a Tensor

        Args:
            w (Tensor): Parameter to initialize
            scale (float): Initialize uniformly in the range [-scale, scale]

        Return:
            None: initialization is done in place
        """
        w.data.uniform_(-scale, scale)

    @staticmethod
    def dim(dim):
        """
        Add any additional dimensions necessary for the manifold

        Args:
            dim (int): dimension specified by user

        Returns int
        """
        return dim

    def normalize(self, u):
        """
        Perform any type of normalization to a Tensor.  Examples include
        fixing a vector to the Hyperboloid (lorentz model) or restricting
        the norm of a vector

        Args:
            u (Tensor): vectors to normalize

        Return:
            Tensor: Normalized tensor
        """
        return u

    @abstractmethod
    def rgrad(self, p, d_p):
        """
        Given the euclidean gradient of ``p`` (``d_p``), computes the
        Riemannian gradient.

        Args:
            p (Tensor): embedding
            d_p (Tensor): euclidean gradient of `p`

        Returns:
            Tensor: Riemannian gradient of `p`
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, u, v):
        """
        Compute the distance between ``u`` and ``v``

        Args:
            u (Tensor): first tensor
            v (Tensor): second tensor

        Returns:
            Tensor: Distance between embeddings ``u`` and ``v``
        """
        raise NotImplementedError

    @abstractmethod
    def expm(self, p, d_p, lr=None, out=None):
        """
        Exponential map for manifold.  Takes a point ``d_p`` in the
        tangent space of ``p`` and maps it on to the manifold

        Args:
            p (Tensor): reference point defining the tangent space
            d_p (Tensor): point in ``p``'s tangent space to be mapped
                on to the manifold

        Returns:
            Tensor: ``d_p`` mapped on to the manifold
        """
        raise NotImplementedError

    @abstractmethod
    def logm(self, x, y):
        """
        Logarithmic map for manifold.  Takes a point ``y`` located on the
        manifold and projects it into the tangent space of ``x``

        Args:
            x (Tensor): reference point defining the tangent space
            y (Tensor): point to be mapped into ``x``'s tangent space
        """
        raise NotImplementedError

    @abstractmethod
    def ptransp(self, x, y, v, ix=None, out=None):
        """
        Parallel transport for manifold.  Assuming ``v`` is in the tangent
        space of ``x``, ``ptransp`` will perform parallel transport into the
        tangent space of ``y``

        Args:
            x (Tensor): starting point
            y (Tensor): end point
            v (Tensor): point in tangent space

        Returns:
            Tensor: embedding parallel transported from the tangent space
            of ``x`` to the tangent space of ``y``
        """
        raise NotImplementedError
