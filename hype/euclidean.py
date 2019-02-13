#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from .manifold import Manifold


class EuclideanManifold(Manifold):
    __slots__ = ["max_norm"]

    def __init__(self, max_norm=1, **kwargs):
        self.max_norm = max_norm

    def normalize(self, u):
        d = u.size(-1)
        u.view(-1, d).renorm_(2, 0, self.max_norm)
        return u

    def distance(self, u, v):
        return (u - v).pow(2).sum(dim=-1)

    def pnorm(self, u, dim=-1):
        return (u * u).sum(dim=dim).sqrt()

    def rgrad(self, p, d_p):
        return d_p

    def expm(self, p, d_p, normalize=False, lr=None, out=None):
        if lr is not None:
            d_p.mul_(-lr)
        if out is None:
            out = p
        out.add_(d_p)
        if normalize:
            self.normalize(out)
        return out

    def logm(self, p, d_p, out=None):
        return p - d_p

    def ptransp(self, p, x, y, v):
        ix, v_ = v._indices().squeeze(), v._values()
        return p.index_copy_(0, ix, v_)


class TranseManifold(EuclideanManifold):
    def __init__(self, dim, *args, **kwargs):
        super(TranseManifold, self).__init__(*args, **kwargs)
        self.r = th.nn.Parameter(th.randn(dim).view(1, dim))

    def distance(self, u, v):
        # batch mode
        if u.dim() == 3:
            r = self.r.unsqueeze(0).expand(v.size(0), v.size(1), self.r.size(1))
        # non batch
        else:
            r = self.r.expand(v.size(0), self.r.size(1))
        return (u - v + r).pow(2).sum(dim=-1)
