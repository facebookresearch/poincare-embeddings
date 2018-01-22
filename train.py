#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import timeit
from torch.utils.data import DataLoader
import gc

_lr_multiplier = 0.1


def train_mp(model, data, optimizer, opt, log, rank, queue):
    try:
        train(model, data, optimizer, opt, log, rank, queue)
    except Exception as err:
        log.exception(err)
        queue.put(None)


def train(model, data, optimizer, opt, log, rank=1, queue=None):
    # setup parallel data loader
    loader = DataLoader(
        data,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=opt.ndproc,
        collate_fn=data.collate
    )

    for epoch in range(opt.epochs):
        epoch_loss = []
        loss = None
        data.burnin = False
        lr = opt.lr
        t_start = timeit.default_timer()
        if epoch < opt.burnin:
            data.burnin = True
            lr = opt.lr * _lr_multiplier
            if rank == 1:
                log.info(f'Burnin: lr={lr}')
        for inputs, targets in loader:
            elapsed = timeit.default_timer() - t_start
            optimizer.zero_grad()
            preds = model(inputs)
            loss = model.loss(preds, targets, size_average=True)
            loss.backward()
            optimizer.step(lr=lr)
            epoch_loss.append(loss.data[0])
        if rank == 1:
            emb = None
            if epoch == (opt.epochs - 1) or epoch % opt.eval_each == (opt.eval_each - 1):
                emb = model
            if queue is not None:
                queue.put(
                    (epoch, elapsed, np.mean(epoch_loss), emb)
                )
            else:
                log.info(
                    'info: {'
                    f'"elapsed": {elapsed}, '
                    f'"loss": {np.mean(epoch_loss)}, '
                    '}'
                )
        gc.collect()
