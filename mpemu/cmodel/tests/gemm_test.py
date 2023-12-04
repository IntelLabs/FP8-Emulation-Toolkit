#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
from mpemu.cmodel import simple

def get_grads(variables):
    return [var.grad.clone() for var in variables]

m=256 
n=512
k=1024

a = torch.rand((k, m), dtype=torch.float32)
b = torch.rand((n, k), dtype=torch.float32)
c = torch.zeros((m, n), dtype=torch.float32)

a64 = a.to(dtype=torch.float64, copy=True)
b64 = b.to(dtype=torch.float64, copy=True)
c64 = c.to(dtype=torch.float64, copy=True)
a32 = a.to(dtype=torch.float32, copy=True)
b32 = b.to(dtype=torch.float32, copy=True)
c32 = c.to(dtype=torch.float32, copy=True)
print("---------->", c.size())
print(a)
print(b)

z = torch.matmul(a64.t(), b64.t())#, out=c64)
z2 = simple.matmul(a.t(), b.t(), out=c)
z3 = torch.matmul(a32.t(), b32.t())#, out=c)
print(z)
print(c-z2)
#print(c32)
#print(z2-z3)
print(z2.size())
print('output 32b: L2 distance : ', torch.dist(z3.to(dtype=torch.float64, copy=True), z, 2))
print('output : L2 distance : ', torch.dist(z2.to(dtype=torch.float64, copy=True), z, 2))
#print('output : L2 distance : ', torch.dist(z2.to(dtype=torch.float64, copy=True), z3, 2))
