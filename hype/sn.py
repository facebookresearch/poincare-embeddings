#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
from torch import nn
from numpy.random import randint
from . import graph
from .graph_dataset import BatchedDataset

model_name = '%s_dim%d'


class Embedding(graph.Embedding):
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__(size, dim, manifold, sparse)
        self.lossfn = nn.functional.cross_entropy

    def _forward(self, e):
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist(s, o).squeeze(-1)
        return -dists

    def loss(self, preds, targets, weight=None, size_average=True):
        return self.lossfn(preds, targets)


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(graph.Dataset):
    def __getitem__(self, i):
        t, h = self.idx[i]
        negs = set()
        ntries = 0
        nnegs = int(self.nnegatives())
        if t not in self._weights:
            negs.add(t)
        else:
            while ntries < self.max_tries and len(negs) < nnegs:
                if self.burnin:
                    n = randint(0, len(self.unigram_table))
                    n = int(self.unigram_table[n])
                else:
                    n = randint(0, len(self.objects))
                if (n not in self._weights[t]) or \
                        (self._weights[t][n] < self._weights[t][h]):
                    negs.add(n)
                ntries += 1
        if len(negs) == 0:
            negs.add(t)
        ix = [t, h] + list(negs)
        while len(ix) < nnegs + 2:
            ix.append(ix[randint(2, len(ix))])
        return th.LongTensor(ix).view(1, len(ix)), th.zeros(1).long()


def initialize(manifold, opt, idx, objects, weights, sparse=True):
    conf = []
    mname = model_name % (opt.manifold, opt.dim)
    data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
        opt.ndproc, opt.burnin > 0, opt.dampening)
    model = Embedding(
        len(data.objects),
        opt.dim,
        manifold,
        sparse=sparse,
    )
    data.objects = objects
    return model, data, mname, conf
