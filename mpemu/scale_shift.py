#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch

class ScaleShift(torch.nn.Module):
    def __init__(self, num_features):
        super(ScaleShift, self).__init__()
        self.num_features = num_features
        self.weight = torch.nn.Parameter(torch.Tensor(num_features))
        self.bias  = torch.nn.Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self,input):
        # ASSUMPTION : First dimension is batch_size
        input_t = input.transpose(1,-1)
        input_t = input_t * self.weight + self.bias
        output = input_t.transpose(1,-1)

        return output

    # For call to repr(). Prints object's information
    def __repr__(self):
       return 'ScaleShift({})'.format(self.num_features)

    @staticmethod
    def generate_from_batchnorm(module: torch.nn.BatchNorm2d):
        """
            Helper function for converting Batchnorm2d to ScaleShift
        """
        # BN stat_dict
        bn_state_dict = module.state_dict()

        # BatchNorm parameters
        num_features = module.num_features
        eps   = module.eps
        rmean = bn_state_dict['running_mean']
        rvar  = bn_state_dict['running_var']
        gamma = bn_state_dict['weight']
        beta  = bn_state_dict['bias']

        # Creating ScaleShift module
        ss_module = ScaleShift(num_features)
        with torch.no_grad():
            denom = torch.sqrt(rvar + eps)
            scale = gamma.div(denom)
            shift = beta - gamma.mul(rmean).div(denom)

            ss_module.weight.copy_(scale)
            ss_module.bias.copy_(shift)

        return ss_module

def replace_batchnorms_with_scaleshifts(model):
    """
        Given a model, replace all BatchNorm2d layers with ScaleShift layers.
    """
    if isinstance(model, torch.nn.BatchNorm2d):
        return ScaleShift.generate_from_batchnorm(model)
    for name, module in model.named_children():
        module = replace_batchnorms_with_scaleshifts(module)
        setattr(model, name, module)
    return model


if __name__ == '__main__':
    #### Testing ScaleShift layer ####
    num_features = 2
    ss =  ScaleShift(num_features)
    ss.weight.data = ss.weight.data * torch.arange(1,num_features+1, dtype=ss.weight.dtype)
    ss.bias.data = torch.arange(1,num_features+1, dtype=ss.weight.dtype)

    input = torch.arange(num_features*2*2).reshape(1,num_features,2,2)
    print("Input")
    print(input)
    print("Scales")
    print(ss.weight.data)
    print("Shifts")
    print(ss.bias.data)

    output = ss(input)
    print("Output")
    print(output)

    #### Testing BN replacement with ScaleShift layer ####
    import torchvision

    model = torchvision.models.__dict__["resnet50"](pretrained=True)
    model.eval()

    # Original model
    input = torch.randn(1,3,224,224)
    output = model(input)

    # Replacing BNs
    model = replace_batchnorms_with_scaleshifts(model)
    output_s = model(input)

    print(torch.norm(output-output_s))
    print(output.flatten()[:10])
    print(output_s.flatten()[:10])
