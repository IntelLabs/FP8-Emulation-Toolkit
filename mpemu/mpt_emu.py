#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

from collections import OrderedDict
import torch
import os
from .qutils import TensorQuantConfig, ModuleQuantConfig
from .qutils import get_or_update_model_quant_config_dict
from .qutils import reset_quantization_setup,add_quantization_hooks
from .qutils import quantize_model_weights,set_quantize_weights_flag
from .e5m2_emu import E5M2Emulator
from .e4m3_emu import E4M3Emulator
from .e3m4_emu import E3M4Emulator
from .bfloat16_emu import Bfloat16Emulator
from .scale_shift import replace_batchnorms_with_scaleshifts
from .module_wrappers import SparseLinear, SparseConv2d 
from .sparse_utils import SparseConfig

'''
Mixed Precision Emulator
'''
class MPTEmulator(object):
    def __init__(self, device='cuda', dtype='fp16', hw_patch='none', pruning_algo='none'):
        super(MPTEmulator, self).__init__()
        self.valid_dtypes = ["fp32", "fp16", "bf16", "e5m2", "e4m3", "e3m4"]
        self.valid_hw_patches = ["none", "simple"]
        self.valid_pruning_methods = ["none", "fine-grained", "unstructured", "adaptive", 'auto']
        # basic checks  
        if dtype.lower() not in self.valid_dtypes: 
            raise RuntimeError("mpt_emulator: the requested data type {} is not supported, use one of the following: {}"
                    .format(dtype, self.valid_dtypes))
        if hw_patch.lower() not in self.valid_hw_patches:
            raise RuntimeError("mpt_emulator: the requested hardware emulation {} is not supported, use one of the following: {}"
                    .format(hw_patch, self.valid_hw_patches))
        if pruning_algo.lower() not in self.valid_pruning_methods: 
            raise RuntimeError("mpt_emulator: the requested pruning method {} is not supported, use one of the following: {}"
                .format(pruning_algo.lower(), self.valid_pruning_methods))

        self.device = device
        self.dtype = dtype.lower()
        self.hw_patch = hw_patch.lower()
        self.pruning_method = pruning_algo.lower()
        self.wt_sparsity = 0.5
        self.grad_sparsity = 0.5

        # emiulator object
        self.emulator = None 
        self.sparse_config = None

    def blacklist_modules(self, list_modules):
        self.emulator.blacklist_modules(list_modules)

    def whitelist_modules(self, list_modules):
        self.emulator.whitelist_modules(list_modules)

    def optimizer_step(self, optimizer):
        self.emulator.optimizer_step(optimizer)
        if self.sparse_config != None:
            self.sparse_config.optimizer_step(optimizer)

    def update_global_steps(self, global_steps):
        self.emulator.global_steps = global_steps

    def set_master_param_precision(self, master_params):
        self.emulator.set_master_param_precision(master_params)

    def set_embedding_precision(self, emb_precision, emb_norm=False):
        self.emulator.set_embedding_precision(emb_precision, emb_norm)

    def enable_tensor_stats(self, summary_writer=None):
        self.emulator.enable_tensor_stats(summary_writer)

    def set_tensor_bindump_schedule(self, list_bindump_schedule):
        self.emulator.set_tensor_bindump_schedule(list_bindump_schedule)

    def enable_tensorboard_logging(self, summary_writer=None):
        self.emulator.enable_tensor_stats(summary_writer)

    def set_target_sparsity_weight(self, wt_sparsity=0.5):
        if self.sparse_config != None:
            self.wt_sparsity = float(wt_sparsity)
            self.sparse_config.weight_factor = self.wt_sparsity
            print("mpt-emulator: weight sparsity has been set to : {}".format(self.sparse_config.weight_factor))
        else:
            print("mpt-emulator: set_target_sparsity_weight has no effect; sparse training is not enabled")

    def set_target_sparsity_gradient(self, grad_sparsity=0.5):
        if self.sparse_config != None:
            self.grad_sparsity = float(grad_sparsity)
            self.sparse_config.outgrad_factor = self.grad_sparsity
            print("mpt-emulator: gradient sparsity has been set to : {}".format(self.sparse_config.outgrad_factor))
        else:
            print("mpt-emulator: set_target_sparsity_gradient has no effect; sparse training is not enabled")

    def fuse_bnlayers_and_quantize_model(self, model):
        return self.emulator.fuse_layers_and_quantize_model(model)

    def set_default_inference_qconfig(self):
        self.emulator.set_default_inference_qconfig()

    def __repr__(self):
        train_infer = "inference"
        if self.is_training :
            train_infer = "training"
        return "[Configured to run {} on {}, using AMP: {}]".format(str(train_infer), self.device, str(self.using_apex))

def rewrite_model_with_adasparse_ops(model, exempt_layers):
    list_exempt_layers = exempt_layers.copy()
    if isinstance(model, torch.nn.Conv2d):
        return SparseConv2d(model.in_channels, model.out_channels, model.kernel_size, 
                stride=model.stride, padding=model.padding, dilation=model.dilation, groups=model.groups, 
                bias=True if model.bias!= None else False).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype) 
    elif isinstance(model, torch.nn.Linear):
        return SparseLinear(model.in_features, model.out_features, 
                bias=True if model.bias!= None else False).to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype) 

    if list_exempt_layers is None:
        for name, module in model.named_children():
            module = rewrite_model_with_adasparse_ops(module, None)
            setattr(model, name, module)
    else : 
        for name, module in model.named_children():
            if name in list_exempt_layers:
                setattr(model, name, module)
                list_exempt_layers.remove(name)
            else:
                module = rewrite_model_with_adasparse_ops(module, list_exempt_layers)
                setattr(model, name, module)
    return model

def initialize(model, optimizer, dtype='fp16', hw_patch='none', pruning_algo='none',
        list_exempt_layers=None, list_layers_output_fused=None, device="cuda", verbose=False ):
    if model is None: #or optimizer is None:
        raise RuntimeError("mpt_emulator: Undefined model and optimizer, call this after model and optimizer are initilized.")
    if dtype == 'fp16' and device != 'cuda':
        raise RuntimeError("mpt_emulator: the requested data type {} is not supported on {}.".format(dtype, device))
    if device == 'cuda' and hw_patch.lower() != 'none':
        raise RuntimeError("mpt_emulator: HW patching ops is only alowed on 'cpu' device.")

    mpt = MPTEmulator(device=device, dtype=dtype, hw_patch=hw_patch, pruning_algo=pruning_algo) 

    if mpt.pruning_method == 'adaptive':
        model = rewrite_model_with_adasparse_ops(model, list_exempt_layers)
        print("mpt_emulator: Adaptive pruning method enabled; Adaptive (weights) only!")
    elif mpt.pruning_method == 'unstructured':
        mpt.sparse_config = SparseConfig ()
        mpt.sparse_config.weight  = True
        mpt.sparse_config.outgrad = True
        mpt.sparse_config.weight_factor  = mpt.wt_sparsity
        mpt.sparse_config.outgrad_factor = mpt.grad_sparsity
        mpt.sparse_config.print_stats = False
        print("mpt_emulator: Unstructured pruning method enabled; TopK(weights : {}), Stochastic(gradients : {})."
                .format(mpt.sparse_config.weight_factor, mpt.sparse_config.outgrad_factor))
    elif mpt.pruning_method == 'auto':
        model = rewrite_model_with_adasparse_ops(model, list_exempt_layers)
        mpt.sparse_config = SparseConfig ()
        mpt.sparse_config.outgrad = True
        mpt.sparse_config.outgrad_factor = mpt.grad_sparsity
        mpt.sparse_config.print_stats = False
        print("mpt_emulator: Auto pruning method enabled; Adaptive(weights), Stochastic(gradients : {})."
                .format(mpt.sparse_config.outgrad_factor))

    if dtype.upper() == 'E5M2':
        mpt.emulator = E5M2Emulator(model, optimizer, mpt.sparse_config, device=device, verbose=verbose)
        mpt.emulator.set_master_param_precision("fp16")

    mpt.emulator.enable_hw_patching(mpt.hw_patch.upper())
    mpt.emulator.prepare_model(model, list_exempt_layers, list_layers_output_fused)
    if mpt.emulator.patch_ops == True and len(mpt.emulator.list_unpatched):  
        print("mpt_emulator: Following layers are not HW_PATCH'ed because thier dimensions do not match the hardware : {} ".format(mpt.emulator.list_unpatched))

    if verbose :
        mpt.emulator.print_config()

    return model, mpt

def quantize_model(model, optimizer=None, dtype="none", hw_patch="none", fuse_bn=False,
        list_exempt_layers=None, list_layers_output_fused=None, device="cuda", verbose=False ):
    if model is None :
        raise RuntimeError("mpt_emulator: Undefined model , call this after model is initilized.")
    if dtype == 'fp16' and device != 'cuda':
        raise RuntimeError("mpt_emulator: the requested data type {} is not supported on {}.".format(dtype, device))
    if device == 'cuda' and hw_patch.lower() != 'none':
        raise RuntimeError("mpt_emulator: HW patching ops is only alowed on 'cpu' device.")

    mpt = MPTEmulator(device=device, dtype=dtype, hw_patch=hw_patch) 
    if fuse_bn :
        model = mpt.fuse_bnlayers_and_quantize_model(model)

    if dtype.upper() == 'E5M2':
        mpt.emulator = E5M2Emulator(model, optimizer, None, device=device, verbose=verbose)
        mpt.set_default_inference_qconfig()
    elif dtype.upper() == 'E4M3':
        mpt.emulator = E4M3Emulator(model, optimizer, None, device=device, verbose=verbose)
        mpt.set_default_inference_qconfig()
    elif dtype.upper() == 'E3M4':
        mpt.emulator = E3M4Emulator(model, optimizer, None, device=device, verbose=verbose)
        mpt.set_default_inference_qconfig()

    mpt.emulator.enable_hw_patching(mpt.hw_patch.upper())
    mpt.emulator.prepare_model(model, list_exempt_layers, list_layers_output_fused)
    if mpt.emulator.patch_ops == True and len(mpt.emulator.list_unpatched):  
        print("mpt_emulator: Following layers are not HW_PATCH'ed because thier dimensions do not match the hardware : {} ".format(mpt.emulator.list_unpatched))

    if verbose :
        mpt.emulator.print_config()

    return model, mpt
