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

class EltwiseAdd(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseAdd, self).__init__()

        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res

    def extra_repr(self) -> str:
        return 'inplace={}'.format(self.inplace)

class EltwiseMul(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseMult, self).__init__()
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res
    def extra_repr(self) -> str:
        return 'inplace={}'.format(self.inplace)


class EltwiseDiv(nn.Module):
    def __init__(self, inplace=False):
        super(EltwiseDiv, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor, y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)
    def extra_repr(self) -> str:
        return 'inplace={}'.format(self.inplace)

