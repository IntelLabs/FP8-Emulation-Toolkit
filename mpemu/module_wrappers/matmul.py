#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import torch.nn as nn

class Matmul(nn.Module):
    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.matmul(a, b)

class BatchMatmul(nn.Module):
    def __init__(self):
        super(BatchMatmul, self).__init__()

    def forward(self, a: torch.Tensor, b:torch.Tensor):
        return torch.bmm(a, b)

class AddMatmul(nn.Module):
    def __init__(self):
        super(AddMatmul, self).__init__()

    def forward(self, input:torch.Tensor, mat1: torch.Tensor, mat2:torch.Tensor, beta=1, alpha=1):
        return torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
