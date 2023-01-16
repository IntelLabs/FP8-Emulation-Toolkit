#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import math
import torch 
import torch.nn as nn

"""
Function for activation binarization
"""
class WeightMaskStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.threshold = nn.Parameter(torch.Tensor(out_features))
        self.weight_mask = WeightMaskStep.apply
        #self.mask = None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0)
    
    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight-threshold
        mask = self.weight_mask(abs_weight)
        ratio = torch.sum(mask) / mask.numel()
        #print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0)
            abs_weight = torch.abs(self.weight)
            threshold = self.threshold.view(abs_weight.shape[0], -1)
            abs_weight = abs_weight-threshold
            mask = self.weight_mask(abs_weight)
        masked_weight = self.weight * mask 
        output = torch.nn.functional.linear(input, masked_weight, self.bias)
        return output
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
class SparseConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_c 
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        ## define weight 
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(out_c))
        self.weight_mask = WeightMaskStep.apply
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        weight_shape = self.weight.shape 
        threshold = self.threshold.view(weight_shape[0], -1)
        weight = torch.abs(self.weight)
        weight = weight.view(weight_shape[0], -1)
        weight = weight - threshold
        mask = self.weight_mask(weight)
        mask = mask.view(weight_shape)
        ratio = torch.sum(mask) / mask.numel()
        # print("threshold {:3f}".format(self.threshold[0]))
        # print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
            threshold = self.threshold.view(weight_shape[0], -1)
            weight = torch.abs(self.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = self.weight_mask(weight)
            mask = mask.view(weight_shape)
        masked_weight = self.weight * mask

        conv_out = torch.nn.functional.conv2d(x, masked_weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(SparseConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'
