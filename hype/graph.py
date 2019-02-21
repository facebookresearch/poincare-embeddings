#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict as ddict
import pandas
import numpy as np
from numpy.random import choice
import torch as th
from torch import nn
from torch.utils.data import Dataset as DS
from sklearn.metrics import average_precision_score
from multiprocessing.pool import ThreadPool
from functools import partial
import h5py
from tqdm import tqdm


def load_adjacency_matrix(path, format='hdf5', symmetrize=False):
    if format == 'hdf5':
        with h5py.File(path, 'r') as hf:
            return {
                'ids': hf['ids'].value.astype('int'),
                'neighbors': hf['neighbors'].value.astype('int'),
                'offsets': hf['offsets'].value.astype('int'),
                'weights': hf['weights'].value.astype('float'),
                'objects': hf['objects'].value
            }
    elif format == 'csv':
        df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')

        if symmetrize:
            rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
            df = pandas.concat([df, rev])

        idmap = {}
        idlist = []

        def convert(id):
            if id not in idmap:
                idmap[id] = len(idlist)
                idlist.append(id)
            return idmap[id]
        df.loc[:, 'id1'] = df['id1'].apply(convert)
        df.loc[:, 'id2'] = df['id2'].apply(convert)

        groups = df.groupby('id1').apply(lambda x: x.sort_values(by='id2'))
        counts = df.groupby('id1').id2.size()

        ids = groups.index.levels[0].values
        offsets = counts.loc[ids].values
        offsets[1:] = np.cumsum(offsets)[:-1]
        offsets[0] = 0
        neighbors = groups['id2'].values
        weights = groups['weight'].values
        return {
            'ids' : ids.astype('int'),
            'offsets' : offsets.astype('int'),
            'neighbors': neighbors.astype('int'),
            'weights': weights.astype('float'),
            'objects': np.array(idlist)
        }
    else:
        raise RuntimeError(f'Unsupported file format {format}')


def load_edge_list(path, symmetrize=False):
    """
    Load an edgelist dataset in CSV format.  The CSV file must have at least
    3 columns: ``id1``, ``id2``, and ``weight``.  If the dataset is directed,
    then it is assumed that ``id2`` is the parent of ``id1``.

    Args:
        path (str): path to the CSV file
        symmetrize (bool): If set to ``True``, then for every edge ``A -> B``,
            we create a symmetric edge ``B -> A``

    Return:
        Tuple[np.ndarray[ndim=2], list[str], np.ndarray[ndim=1]]: 
            A tuple containiner: ``idx`` an array of edges, ``objects`` a 
            list of the unique objects in the graph, and ``weights`` an array
            the same length of ``idx`` containing the weights of each edge
    """
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights


class Embedding(nn.Module):
    """
    Base class for Embedding models

    Args:
        size (int): number of embeddings
        dim (int): dimension of the embeddings
        manifold (Manifold): which manifold to use
        sparse (bool): whether or not to use sparse gradients
    """
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__()
        self.dim = dim
        self.nobjects = size
        self.manifold = manifold
        self.lt = nn.Embedding(size, dim, sparse=sparse)
        self.dist = manifold.distance
        self.pre_hook = None
        self.post_hook = None
        self.init_weights(manifold)

    def init_weights(self, manifold, scale=1e-4):
        manifold.init_weights(self.lt.weight, scale)

    def forward(self, inputs):
        e = self.lt(inputs)
        with th.no_grad():
            e = self.manifold.normalize(e)
        if self.pre_hook is not None:
            e = self.pre_hook(e)
        fval = self._forward(e)
        return fval

    def embedding(self):
        return list(self.lt.parameters())[0].data.cpu().numpy()

    def optim_params(self, manifold):
        return [{
            'params': self.lt.parameters(),
            'rgrad': manifold.rgrad,
            'expm': manifold.expm,
            'logm': manifold.logm,
            'ptransp': manifold.ptransp,
        }, ]


# This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(DS):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75

    def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert weights.ndim == 1
        assert len(idx) == len(weights)
        assert nnegs >= 0
        assert unigram_size >= 0

        print('Indexing data')
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += weights[i]
            self._weights[t][h] += weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(objects) > nents, f'Number of objects do no match'

        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = choice(
                len(objects),
                size=int(unigram_size),
                p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return th.cat(inputs, 0), th.cat(targets, 0)


# This function is now deprecated in favor of eval_reconstruction
def eval_reconstruction_slow(adj, lt, distfn):
    ranks = []
    ap_scores = []

    for s, s_types in adj.items():
        s_e = lt[s].expand_as(lt)
        _dists = distfn(s_e, lt).data.cpu().numpy().flatten()
        _dists[s] = 1e+12
        _labels = np.zeros(lt.size(0))
        _dists_masked = _dists.copy()
        _ranks = []
        for o in s_types:
            _dists_masked[o] = np.Inf
            _labels[o] = 1
        for o in s_types:
            d = _dists_masked.copy()
            d[o] = _dists[o]
            r = np.argsort(d)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
        ap_scores.append(
            average_precision_score(_labels, -_dists)
        )
    return np.mean(ranks), np.mean(ap_scores)


def reconstruction_worker(adj, lt, distfn, objects, progress=False):
    ranksum = nranks = ap_scores = iters = 0
    labels = np.empty(lt.size(0))
    for object in tqdm(objects) if progress else objects:
        labels.fill(0)
        neighbors = np.array(list(adj[object]))
        dists = distfn(lt[None, object], lt)
        dists[object] = 1e12
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.cpu().numpy(), neighbors))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]

        # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
        # As an example, assume the ranks of the neighbors are:
        # 0, 1, 4, 5, 6, 8
        # For each neighbor, we'd like to return the number of non-neighbors
        # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
        # Another way of thinking about it is to return
        # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
        # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
        # Note that we include `N` to account for the source embedding itself
        # always being the nearest neighbor
        ranksum += ranks.sum() - (N * (N - 1) / 2)
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        ap_scores += average_precision_score(labels, -dists.cpu().numpy())
        iters += 1
    return float(ranksum), nranks, ap_scores, iters


def eval_reconstruction(adj, lt, distfn, workers=1, progress=False):
    '''
    Reconstruction evaluation.  Evaluate how well the embedding is able
    to reconstruct the original input graph.  Specifically, for each node,
    we compute all of its nearest neighbors in the embedding space and rank
    them amongst its non-neighbors.

    Args:
        adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
        lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
            dimensionality
        distfn (Callable[[Tensor, Tensor], Tensor]): distance function to use for
            computing nearest neighbors in embedding space
        workers (int): number of workers to use
        progress (bool): display progress bar

    Returns:
        Tuple[float, float]: ``mean_rank``, ``map_rank``
    '''
    objects = np.array(list(adj.keys()))
    if workers > 1:
        with ThreadPool(workers) as pool:
            f = partial(reconstruction_worker, adj, lt, distfn)
            results = pool.map(f, np.array_split(objects, workers))
            results = np.array(results).sum(axis=0).astype(float)
    else:
        results = reconstruction_worker(adj, lt, distfn, objects, progress)
    return float(results[0]) / results[1], float(results[2]) / results[3]
