#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, p='fro', dim=None, keepdim=False):
        super(Norm, self).__init__()
        self.p = p
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor):
        return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)
    def extra_repr(self) -> str:
        return 'p={}, dim={}, keepdim: {}'.format(self.p, self.dim, self.keepdim)

class Mean(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Mean, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor):
        return torch.mean(x, *self.args, **self.kwargs)
    def extra_repr(self) -> str:
        return 'args={}, kwargs={}'.format(self.args, self.kwargs)

