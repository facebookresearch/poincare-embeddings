#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from os.path import join as pjoin
import time
import torch


class LocalCheckpoint(object):
    """
    Module for managing model checkpoints.

    Args:
        path (str): path to save the checkpoint to
        include_in_all (dict): a dictionary of objects to save in every call to
            :func:``save``
        start_fresh (bool): If ``True``, then ignore any existing checkpoint,
        otherwise initialize from previous checkpoint
    """
    def __init__(self, path, include_in_all=None, start_fresh=False):
        self.path = path
        self.start_fresh = start_fresh
        self.include_in_all = {} if include_in_all is None else include_in_all

    def initialize(self, params):
        """
        Initialize the checkpoint.  If ``start_fresh`` is ``True``, then ``params`` is
        returned.  Otherwise if a checkpoint at ``self.path`` exists, the checkpoint
        is loaded and returned

        Args:
            params (dict): checkpoint contents

        Returns:
            dict: Either ``params`` or the contents of the checkpoint stored at
            ``self.path``
        """
        if not self.start_fresh and os.path.isfile(self.path):
            print(f'Loading checkpoint from {self.path}')
            return torch.load(self.path)
        else:
            return params

    def save(self, params, tries=10):
        """
        Save a checkpoint containing ``params`` merged with ``self.include_in_all``

        Args:
            params(dict): data to store in checkpoint.  This is merged with
                anything supplied to ``include_in_all`` in the constructor
            tries(int): number of attempts to try and save the checkpoint.
                If the number of attempts exhausts, then no checkpoint is
                saved
        Returns:
            None
        """
        try:
            torch.save({**self.include_in_all, **params}, self.path)
        except Exception as err:
            if tries > 0:
                print(f'Exception while saving ({err})\nRetrying ({tries})')
                time.sleep(60)
                self.save(params, tries=(tries - 1))
            else:
                print("Giving up on saving...")
