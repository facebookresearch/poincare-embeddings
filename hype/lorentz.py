#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch.autograd import Function
from .common import acosh
from .manifold import Manifold


class LorentzManifold(Manifold):
    """
    Lorentz model of hyperbolic geometry.  This is the manifold used
    in `"Learning Continuous Hierarchies in the Lorentz Model of 
    Hyperbolic Geometry" (Nickel et al., 2018) <https://arxiv.org/abs/1806.03417>`_

    """
    __slots__ = ["eps", "_eps", "norm_clip", "max_norm", "debug"]

    @staticmethod
    def dim(dim):
        """See :func:`~hype.manifold.Manifold.dim`"""
        return dim + 1

    def __init__(self, eps=1e-12, _eps=1e-5, norm_clip=1, max_norm=1e6,
            debug=False, **kwargs):
        self.eps = eps
        self._eps = _eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm
        self.debug = debug

    @staticmethod
    def ldot(u, v, keepdim=False):
        """
        Computes the Lorentzian Scalar Product between ``u`` and ``v``
        
        :math:`\\langle u, v \\rangle_L = -u_0 * v_0 + \\sum_{i=1}^n u_i * v_i`

        Args:
            u (Tensor): embedding
            v (Tensor): embedding

        Returns:
            Tensor
        """
        uv = u * v
        uv.narrow(-1, 0, 1).mul_(-1)
        return th.sum(uv, dim=-1, keepdim=keepdim)

    def to_poincare_ball(self, u):
        x = u.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def distance(self, u, v):
        """
        See :func:`~hype.manifold.Manifold.distance`
    

        :math:`d(u, v) = \\text{acosh}(-\\langle u, v, \\rangle_L)`
        """
        d = -LorentzDot.apply(u, v)
        d.data.clamp_(min=1)
        return acosh(d, self._eps)

    def pnorm(self, u):
        return th.sqrt(th.sum(th.pow(self.to_poincare_ball(u), 2), dim=-1))

    def normalize(self, w):
        """See :func:`~hype.manifold.Manifold.normalize`"""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed.view(-1, d).renorm_(p=2, dim=0, maxnorm=self.max_norm)
        tmp = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w

    def normalize_tan(self, x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp.sqrt_().clamp_(min=self._eps)
        v_all.narrow(1, 0, 1).copy_(xv / tmp)
        return v_all

    def init_weights(self, w, irange=1e-5):
        """
        Same as :func:`~hype.manifold.Manifold.init_weights`, but also fixes the 
        normalized embeddings to the hyperboloid
        """
        w.data.uniform_(-irange, irange)
        w.data.copy_(self.normalize(w.data))

    def rgrad(self, p, d_p):
        """See :func:`~hype.manifold.Manifold.rgrad`"""
        if d_p.is_sparse:
            u = d_p._values()
            x = p.index_select(0, d_p._indices().squeeze())
        else:
            u = d_p
            x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def expm(self, p, d_p, lr=None, out=None, normalize=False):
        """
        See :func:`~hype.manifold.Manifold.expm`
        

        :math:`exp_p(d_p) = \\text{cosh}(||d_p||_L)p + \\text{sinh}(||d_p||)
        \\frac{d_p}{||d_p||_L}`
        """
        if out is None:
            out = p
        if d_p.is_sparse:
            ix, d_val = d_p._indices().squeeze(), d_p._values()
            # This pulls `ix` out of the original embedding table, which could
            # be in a corrupted state.  normalize it to fix it back to the
            # surface of the hyperboloid...
            # TODO: we should only do the normalize if we know that we are
            # training with multiple threads, otherwise this is a bit wasteful
            p_val = self.normalize(p.index_select(0, ix))
            ldv = self.ldot(d_val, d_val, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p_val).addcdiv_(th.sinh(t) * d_val, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.index_copy_(0, ix, newp)
        else:
            if lr is not None:
                d_p.narrow(-1, 0, 1).mul_(-1)
                d_p.addcmul_((self.ldot(p, d_p, keepdim=True)).expand_as(p), p)
                d_p.mul_(-lr)
            ldv = self.ldot(d_p, d_p, keepdim=True)
            if self.debug:
                assert all(ldv > 0), "Tangent norm must be greater 0"
                assert all(ldv == ldv), "Tangent norm includes NaNs"
            nd_p = ldv.clamp_(min=0).sqrt_()
            t = th.clamp(nd_p, max=self.norm_clip)
            nd_p.clamp_(min=self.eps)
            newp = (th.cosh(t) * p).addcdiv_(th.sinh(t) * d_p, nd_p)
            if normalize:
                newp = self.normalize(newp)
            p.copy_(newp)

    def logm(self, x, y):
        """See :func:`~hype.manifold.Manifold.logm`"""
        xy = th.clamp(self.ldot(x, y).unsqueeze(-1), max=-1)
        v = acosh(-xy, self.eps).div_(
            th.clamp(th.sqrt(xy * xy - 1), min=self._eps)
        ) * th.addcmul(y, xy, x)
        return self.normalize_tan(x, v)

    def ptransp(self, x, y, v, ix=None, out=None):
        """See :func:`~hype.manifold.Manifold.ptransp`"""
        if ix is not None:
            v_ = v
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        elif v.is_sparse:
            ix, v_ = v._indices().squeeze(), v._values()
            x_ = x.index_select(0, ix)
            y_ = y.index_select(0, ix)
        else:
            raise NotImplementedError
        xy = self.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        if out is None:
            return vnew
        else:
            out.index_copy_(0, ix, vnew)


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u
