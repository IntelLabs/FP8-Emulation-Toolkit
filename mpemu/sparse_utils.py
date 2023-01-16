#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import os
import sys

class SparseConfig(object):
    def __init__(self, weight=False, ingrad=False, outgrad=False, wtgrad=False):
        self.weight  = weight
        self.ingrad  = ingrad
        self.outgrad = outgrad
        self.wtgrad  = wtgrad
        self.print_stats = False

        self.global_step = 0
        self.alpha_window = 50
        self.weight_alpha = 65504.0 
        self.weight_factor = 0.0
        self.outgrad_alpha = 65504.0 
        self.outgrad_factor = 0.0

    def __repr__(self):
        return "[weight: {}, outgrad: {}, alpha_window : {}, weight_sparsity: {}, outgrd_sparsity :{} ]".format(
                self.weight, self.outgrad, self.alpha_window, self.weight_factor, self.outgrad_factor)

    def sparsify_ingrad_tensor(self, in_grad):
        return in_grad

    def sparsify_outgrad_tensor(self, out_grad):
        if self.global_step != 0 and (self.global_step%self.alpha_window) == 0 :
            self.outgrad_alpha = Stochastic_Pruning_Threshold(out_grad, self.outgrad_factor)

        out_grad.data = Stochastic_Pruning(out_grad.data, self.outgrad_alpha)
        return out_grad

    def sparsify_weight_tensor(self, weight):
        sample_factor = 0.1
        if self.global_step != 0 and (self.global_step%self.alpha_window) == 0 :
            self.weight_alpha = Topk_Threshold_Sampled(weight.data, self.weight_factor, sample_factor)

        weight.data = Topk_Pruning(weight.data, self.weight_alpha)
        return weight

    def sparsify_wtgrad_tensor(self, wt_grad):
        return wt_grad

    def optimizer_step (self, optimizer):
        '''
        This routine should handle any sparsity related operations that needs to performed inside the optimier
        '''
        return

from scipy.optimize import root_scalar
import pdb

def print_sparse_stats (module, grad_input, grad_output) :
    for in_grad, out_grad in zip(grad_input, grad_output):
        wt_sp = 0.0
        if type(module) in [torch.nn.Conv2d, torch.nn.Linear] :
            wt_sp = 1 - (torch.count_nonzero(module.weight.data)/module.weight.data.numel()).item()
        grad_sp = 1 - (torch.count_nonzero(out_grad.data)/out_grad.data.numel()).item()
        print ("{} , weight_sparsity : {}, alpha_weight : {}, grad_sparsity : {}".format(module.name, wt_sp, module.qconfig.weight_ALPHA, grad_sp))

def Stochastic_Pruning(X, ALPHA):
    rand = ALPHA * torch.rand(X.shape, device=X.device,dtype=X.dtype)
    X_abs = X.abs()
    X = torch.where(X_abs < ALPHA, ALPHA * torch.sign(X), X)
    X = torch.where(X_abs < rand, torch.tensor([0], device=X.device, dtype=X.dtype), X)
    del X_abs
    return X

def Stochastic_Pruning_Threshold(X, Sparsity):
    X = X.abs()
    X_sp = 1 - (torch.count_nonzero(X)/X.numel()).item()
    if Sparsity <= X_sp:
        return X
    Target_sparsity = Sparsity - X_sp

    Y = torch.log(X[X!=0.0])
    mu = torch.mean(Y, dtype=torch.float)
    sigma = torch.std(Y)
    del Y
    guess = torch.tensor([1], device=X.device, dtype=X.dtype)
    bracket = [torch.exp(torch.tensor([-9], device=X.device, dtype=torch.float)),
    torch.exp(torch.tensor([5], device=X.device, dtype=torch.float))]
    sol = root_scalar(equationStochastic, x0=guess, bracket=bracket, args=(Sparsity, torch.tensor([0.0], device=X.device), sigma))
    ALPHA = torch.tensor([sol.root], device=X.device, dtype=X.dtype) 
    return torch.exp(torch.log(ALPHA) + mu)

def Topk_Pruning(weight, alpha):
    Weight_Mask = torch.where(torch.abs(weight) < alpha, torch.tensor([0], device=weight.device, dtype=torch.short),\
                                                        torch.tensor([1], device=weight.device, dtype=torch.short))
    weight.mul_(Weight_Mask.to(device=weight.device))
    del Weight_Mask
    return weight

def Topk_Threshold_Sampled(weight, Sparsity_Weight, sample_factor):
    Total_Weight_El = weight.numel()
    sampled_El = int(Total_Weight_El*sample_factor)
    sampled_id = torch.randint(0, Total_Weight_El, (1,sampled_El))
    X_sampled = torch.abs(weight.view(-1))[sampled_id]
    #pdb.set_trace()
    topk_count_sample = int(sampled_El*(1-Sparsity_Weight))
    __ , topk_w_id_sampled = torch.topk(X_sampled, topk_count_sample)
    alpha = X_sampled[0,topk_w_id_sampled[0,topk_count_sample-1]]
    return alpha

def equationStochastic(alpha, sparsity, mu, sigma):
    sqrt2 = torch.sqrt(torch.tensor([2], device="cuda", dtype=torch.float))
    alpha = torch.tensor([alpha], device="cuda", dtype=torch.float)
    pt1 = torch.exp((sigma**2)/2) * torch.erf(sigma/sqrt2 - torch.log(alpha/torch.exp(mu))/(sqrt2 * sigma))
    pt2 = alpha/torch.exp(mu) * torch.erf(torch.log(alpha/torch.exp(mu))/(sqrt2 * sigma))
    pt3 = torch.exp((sigma**2)/2)
    return 0.5 - sparsity + torch.exp(mu)/(2*alpha) * (pt1 + pt2 - pt3)
