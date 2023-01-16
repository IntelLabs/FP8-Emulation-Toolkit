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
from .scale_shift import replace_batchnorms_with_scaleshifts
from .module_wrappers import BatchMatmul,Matmul,AddMatmul,EltwiseMul,EltwiseAdd,EltwiseDiv
from .module_wrappers import SparseConv2d,SparseLinear
from torch.quantization.fuser_method_mappings import fuse_conv_bn
from torch.quantization.fuser_method_mappings import fuse_conv_bn_relu

'''
Bfloat8 Mixed Precision Emulator
'''
class Bfloat16Emulator(object):
    def __init__(self, model, optimizer, sparse_config=None, device="cuda", verbose=False, tensor_stats=False):
        super(Bfloat16Emulator, self).__init__()
        self.whitelist = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag]
        self.whitelist += [Matmul, BatchMatmul, AddMatmul]
        self.whitelist += [EltwiseAdd, EltwiseMul, EltwiseDiv]
        self.whitelist += [SparseConv2d, SparseLinear]
        self.blacklist = []
        self.list_unpatched = []
        self.list_exempt_layers = None
        self.list_layers_output_fused = None
        self.using_apex = False
        self.use_fp16_master = False
        self.use_bf16_master = False
        self.use_bf16_emb = False
        self.device = device
        self.model = model
        self.global_steps = 0
        self.emb_norm = False
        self.is_training = model.training
        self.patch_ops = False
        self.patch_impl = "NONE"
        self.patchlist = ["TMUL", "DPAS", "DPAS_BNA4", "DPAS_LA", "DPAS_LA_BNA4", "SSDPAS", "SSDPAS_LA", "XE_PVC"]
        self.patchlist += ["DPAS_BB4FP16", "DPAS_BB4FP12", "DPAS_BB8FP16", "DPAS_BB8FP12"] 
        self.sparse_config = sparse_config
        # default configuration
        self.mod_qconfig = None
        self.model_qconfig_dict = None #OrderedDict()
        self.data_emulation = True
        self.emb_qconfig    = None #TensorQuantConfig("bfloat16", "rne")
        self.wt_qconfig     = TensorQuantConfig("bfloat16", "rne")
        self.iact_qconfig   = TensorQuantConfig("bfloat16", "rne")
        self.oact_qconfig   = TensorQuantConfig("bfloat16", "rne")
        self.igrad_qconfig  = TensorQuantConfig("bfloat16", "rne")
        self.ograd_qconfig  = TensorQuantConfig("bfloat16", "rne")
        self.wtgrad_qconfig = TensorQuantConfig("bfloat16", "rne")
        self.hook_handles = None
        self.bin_dump_enable = None
        self.bin_dump_disable = None
        self.verbose = verbose
        self.tensor_stats = tensor_stats
        self.tb_writer = None
        if self.is_training == True and optimizer is not None:
            if hasattr(optimizer, "_amp_stash"):
                self.using_apex = True
            else :
                if torch.cuda.is_available():
                    raise RuntimeError("Failed to initialize, optimizer did not pass through amp.initialize. {}".format(\
                    "Call amp.initilize(model, optimizer, opt_level='O2') before caling this interface."))
                else :
                    raise RuntimeError("Failed to initialize, optimizer did not pass through amp.initialize. {} {}".format(\
                    "Call amp.initilize(model, optimizer, opt_level='O0') before caling this interface.",\
                    "Install Apex with 'apex_cpu.patch' if you running this on CPU. Please refer to README.md for instructions."))


    def blacklist_modules(self, list_modules) :
        if self.verbose:
            print("Current module list {}".format(self.whitelist))
            print("Blacklist-ing modules {}".format(list_modules))
        for module in list_modules :
            self.blacklist.append(module)
            self.whitelist.remove(module)
        self.create_or_update_hooks()
        if self.verbose:
            print("Updated module list {}".format(self.whitelist))
            self.print_config()

    def whitelist_modules(self, list_modules) :
        if self.verbose:
            print("Current module list {}".format(self.whitelist))
            print("Whitelist-ing modules {}".format(list_modules))
        for module in list_modules :
            self.whitelist.append(module)
            self.blacklist.remove(module)
        self.create_or_update_hooks()
        if self.verbose:
            print("Updated module list {}".format(self.whitelist))
            self.print_config()

    def set_default_inference_qconfig(self):
        self.emb_qconfig    = TensorQuantConfig("bfloat16", "rne")
        self.wt_qconfig     = TensorQuantConfig("bfloat16", "rne")
        self.iact_qconfig   = TensorQuantConfig("bfloat16", "rne")
        self.oact_qconfig   = None 
        self.igrad_qconfig  = None 
        self.ograd_qconfig  = None
        self.wtgrad_qconfig = None

    def _check_master_weights(self, optimizer):
        '''
        for name in ("_lazy_init_maybe_master_weights",
                     "_master_params_to_model_params"):
            if hasattr(optimizer, name):
                return True
        '''
        if hasattr(optimizer, "_master_params_to_model_params"):
            return True
        return False

    def update_fp16_master_params(self, optimizer) :
        if self.device == 'cuda' :
            from .pytquant.cuda import quantemu_cuda
            for name, param in self.model.named_parameters():
                if self.use_bf16_emb and "embeddings" in name:
                    if self.emb_norm :
                        param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True, self.emb_norm, param.size()[1])
                    else :
                        param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True)
                else:
                    param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)

                if self.tensor_stats and (self.global_steps % 1000) == 0:
                    self.tb_writer.add_histogram(name, param, global_step=self.global_steps)
                    self.tb_writer.add_scalar(name+"_absmax", torch.max(torch.abs(param)), global_step=self.global_steps)
                    self.tb_writer.add_scalar(name+"_absmin", torch.min(torch.abs(param)), global_step=self.global_steps)
        else :
            from .pytquant.cpp import quantemu_cpp
            for name, param in self.model.named_parameters():
                if self.use_bf16_emb and "embeddings" in name:
                    if self.emb_norm :
                        param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True, self.emb_norm, param.size()[1])
                    else :
                        param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True)
                else :
                    param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)

                if self.tensor_stats and (self.global_steps % 1000) == 0:
                    self.tb_writer.add_histogram(name, param, global_step=self.global_steps)
                    self.tb_writer.add_scalar(name+"_absmax", torch.max(torch.abs(param)), global_step=self.global_steps)
                    self.tb_writer.add_scalar(name+"_absmin", torch.min(torch.abs(param)), global_step=self.global_steps)

    def update_bf16_master_params(self, optimizer) :
        for name, param in self.model.named_parameters():
            if self.device == 'cuda' :
                from .pytquant.cuda import quantemu_cuda
                if self.use_fp16_emb and "embeddings" in name:
                    if self.emb_norm :
                        param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True, self.emb_norm, param.size()[1])
                    else :
                        param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                else : 
                    param.data = quantemu_cuda.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True)
            else :
                from .pytquant.cpp import quantemu_cpp
                if self.use_fp16_emb and "embeddings" in name:
                    if self.emb_norm :
                        param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True, self.emb_norm, param.size()[1])
                    else :
                        param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                else :
                    param.data = quantemu_cpp.QuantEmuOp.apply(param.data, "BFLOAT16_STOCHASTIC", True)

    def update_master_params(self, optimizer) :
        if self.use_fp16_master:
            self.update_fp16_master_params(optimizer)
        elif self.use_bf16_master:
            self.update_bf16_master_params(optimizer)

    def optimizer_step(self, optimizer):
        optimizer.step()
        self.update_master_params(optimizer)
        self.global_steps = self.global_steps + 1

        if self.tensor_stats and (self.global_steps % 1000) == 0:
            for name,module in self.model.named_modules():
                if type(module) in [torch.nn.Conv2d, torch.nn.Linear]\
                    and name in self.model_qconfig_dict :
                    module.qconfig.global_step = self.global_steps
                    module.qconfig.tb_writer = self.tb_writer
                    module.qconfig.tensor_stats = True 

        if self.tensor_stats and (self.global_steps % 1000) == 1:
             for name,module in self.model.named_modules():
                if type(module) in [torch.nn.Conv2d, torch.nn.Linear]\
                    and name in self.model_qconfig_dict :
                    module.qconfig.tensor_stats = False 
                    module.qconfig.global_step = -1
                    module.qconfig.tb_writer = None 
       
        if self.bin_dump_enable is not None and self.global_steps in self.bin_dump_enable:
            for name,module in self.model.named_modules():
                if type(module) in [torch.nn.Conv2d, torch.nn.Linear]\
                    and name in self.model_qconfig_dict :
                    module.qconfig.bin_dump = True
                    module.qconfig.global_step = self.global_steps
        
        if self.bin_dump_disable is not None and self.global_steps in self.bin_dump_disable:
            for name,module in self.model.named_modules():
                if type(module) in [torch.nn.Conv2d, torch.nn.Linear]\
                    and name in self.model_qconfig_dict :
                    module.qconfig.bin_dump = False
                    module.qconfig.global_step = -1

    def create_or_update_hooks(self, model):
        self.model_qconfig_dict = get_or_update_model_quant_config_dict(model,
                                    self.whitelist, self.mod_qconfig,
                                    model_qconfig_dict=self.model_qconfig_dict,
                                    override=True)

        if self.list_exempt_layers is not None :
            for exempt_layer in self.list_exempt_layers:
                if self.model_qconfig_dict.get(exempt_layer) is not None:
                    self.model_qconfig_dict.pop(exempt_layer)

		# Disable output quantization for these layers,
		# These layers are followed by precision sensitive layers such as SoftMax
        # In the final implementation, sensitive layers are fused with the preceeding layer
        if self.list_layers_output_fused is not None :
            for name,module in model.named_modules():
                if name in self.list_layers_output_fused and name not in self.list_exempt_layers \
                and type(module) in self.whitelist :
                    self.model_qconfig_dict[name].oact_qconfig = None
                    self.model_qconfig_dict[name].ograd_qconfig = None

        # additional handling of HW patching 
        for name,module in model.named_modules():
            if type(module) in [torch.nn.Conv2d] and name in self.model_qconfig_dict:
                if module.in_channels < 64 or module.out_channels < 64:
                    self.model_qconfig_dict[name].patch_ops = False
                    self.model_qconfig_dict[name].patch_impl = "NONE"
                    self.list_unpatched += [name]
 
        # Except for Conv2d, Linear and Embedding module disable quantization on weight and gradient
        for name,module in model.named_modules():
            if type(module) not in [torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag]\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = None
                self.model_qconfig_dict[name].wtgrad_qconfig = None

        for name,module in model.named_modules():
            if ((type(module) == torch.nn.Embedding) or (type(module) == torch.nn.EmbeddingBag))\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = self.emb_qconfig
                self.model_qconfig_dict[name].iact_qconfig = None
                self.model_qconfig_dict[name].igrad_qconfig = None
                self.model_qconfig_dict[name].ograd_qconfig = None
                self.model_qconfig_dict[name].oact_qconfig = None

        for name,module in model.named_modules():
            if type(module) in [BatchMatmul]\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = None
                self.model_qconfig_dict[name].wtgrad_qconfig = None
                self.model_qconfig_dict[name].oact_qconfig = None
                self.model_qconfig_dict[name].ograd_qconfig = None

        reset_quantization_setup(model, self.model_qconfig_dict)
        # Adding hooks for quantizing input.
        self.hook_handles = add_quantization_hooks(model, self.model_qconfig_dict, is_training=self.is_training)
        if not self.is_training :
            quantize_model_weights(model, self.model_qconfig_dict)
            set_quantize_weights_flag(model, self.model_qconfig_dict, False)

    def prepare_model(self, model, list_exempt_layers, list_layers_output_fused):
        mod_qconfig = ModuleQuantConfig(wt_qconfig=self.wt_qconfig,
                                    iact_qconfig=self.iact_qconfig,
                                    oact_qconfig=self.oact_qconfig,
				    igrad_qconfig=self.igrad_qconfig,
				    ograd_qconfig=self.ograd_qconfig,
				    wtgrad_qconfig=self.wtgrad_qconfig,
				    patch_ops=self.patch_ops)
        mod_qconfig.device = self.device
        mod_qconfig.patch_impl = self.patch_impl
        mod_qconfig.sparse_config = self.sparse_config
        self.mod_qconfig = mod_qconfig
        self.list_exempt_layers = list_exempt_layers
        self.list_layers_output_fused = list_layers_output_fused
        self.create_or_update_hooks(model)

    def set_master_param_precision(self, master_params):
        if master_params == 'fp16' or master_params == 'FP16' :
            self.use_fp16_master = True
            self.use_bf16_master = False
            if self.verbose :
                print("bfloat16_emulator: Use float16 master parameters = ", self.use_fp16_master)
        if master_params == 'bf16' or master_params == 'BF16' :
            self.use_bf16_master = True
            self.use_fp16_master = False
            if self.verbose :
                print("bfloat16_emulator: Use bfloat16 master parameters = ", self.use_bf8_master)
 
    def set_embedding_precision(self, emb_precision, emb_norm):
        if emb_precision == 'bf16' or emb_precision == 'BF16' :
            self.use_bf16_emb = True
            if verbose :
                print("bfloat16_emulator: Use bfloat16 embedding table = ", self.use_bf16_emb)
        if emb_norm and self.use_bf16_emb :
            self.emb_norm = emb_norm
            if verbose :
                print("bfloat16_emulator: Using block normalization for embeddings")

    def enable_tensor_stats(self, summary_writer=None):
        if summary_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                from datetime import datetime
                self.tb_writer = SummaryWriter("emulator_logs/"+model.__class__.__name__+"_"+device+"_"+"{:%Y_%m_%d_%H_%M_%S}".format(datetime.now()))
                self.tensor_stats = True
            except ImportError:
                emulator.tensor_stats = False
                emulator.tb_writer = None
                print("Disabled tensor_Stats becuse tensorboard is not installed; Please install tensorboard and relaunch training.")
        else:
            self.tb_writer = summary_writer 
            self.tensor_stats = True

    def set_tensor_bindump_schedule(self, list_bindump_schedule):
        if list_bindump_schedule is not None :
            print("bfloat16_emulator: Enabled binary dump for iterations : {} ".format(list_bindump_schedule))
            self.bin_dump_enable = list_bindump_schedule 
            bindump_steps = torch.tensor(list_bindump_schedule ,dtype=torch.int32)
            next_step = torch.ones(len(list_bindump_schedule), dtype=torch.int32)
            bindump_steps = bindump_steps + next_step
            self.bin_dump_disable = bindump_steps.tolist() 

    def enable_hw_patching(self, patch_ops):
        if patch_ops != 'NONE':
            if patch_ops in self.patchlist :
                self.patch_ops = True
                self.patch_impl = patch_ops
                print("bfloat16_emulator: PyTorch Ops are monkey-patched to use {} kernels : {}".format(self.patch_impl, self.patch_ops))
            else :
                raise RuntimeError("bfloat16_emulator: HW patching is not supported for {}, supported list of options : {}".format(patch_ops, self.patchlist))

    def fuse_layers_and_quantize_model(self, model):
        #if self.is_training :
        if model.training:
            print("Warning : emulator.is_training is set to True, returning the model unchanged")
            return model
        if self.verbose :
            print("Fusing Batchnorm layers and replacing them with scale and shift")

        modules_to_fuse = [ ['conv1', 'bn1', 'relu'], ['submodule.conv', 'submodule.relu']]
        fuse_custom_config_dict = {
                "additional_fuser_method_mapping": {(torch.nn.Conv2d, torch.nn.BatchNorm2d): fuse_conv_bn },
                }
        #fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, fuser_func=fuse_conv_bn(torch.nn.Conv2d, torch.nn.BatchNorm2d))
        #fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, fuse_conv_bn([torch.nn.Conv2d, torch.nn.BatchNorm2d]))
        #fused_model = torch.quantization.fuse_modules(model, modules_to_fuse, fuse_custom_config_dict=fuse_custom_config_dict)
        fused_model = replace_batchnorms_with_scaleshifts(model)
        #reset_quantization_setup(fused_model, self.model_qconfig_dict)
        #add_quantization_hooks(fused_model, self.model_qconfig_dict)
        #quantize_model_weights(model, self.model_qconfig_dict) # added new
        #set_quantize_weights_flag(fused_model, self.model_qconfig_dict, False)
        fused_model = fused_model.to(self.device)
        return fused_model

    def disable_datatype_emulation(self):
        self.data_emulation = False
        self.emb_qconfig    = None 
        self.wt_qconfig     = None
        self.iact_qconfig   = None
        self.oact_qconfig   = None
        self.igrad_qconfig  = None
        self.ograd_qconfig  = None
        self.wtgrad_qconfig = None

    def print_config(self):
        for key in self.model_qconfig_dict:
            print("{} {:40s}".format(self.model_qconfig_dict[key], key))

    def __repr__(self):
        train_infer = "inference"
        if self.is_training :
            train_infer = "training"
        return "[Configured to run {} on {}, using AMP: {}]".format(str(train_infer), self.device, str(self.using_apex))

'''

'''
def prepare_model_for_training(model, optimizer, list_exempt_layers=None, list_layers_output_fused=None,
        master_params='fp32', emb_precision='fp32', emb_norm=False, device="cuda", patch_ops='None', 
        verbose=False, tensor_stats=False, list_bindump_schedule=None ):

    if model is None or optimizer is None:
        raise RuntimeError("bfloat16_emulator: Undefined model and optimizer, call this after model and optimizer are initilized.")

    if device == 'cuda' and patch_ops != 'None':
        raise RuntimeError("bfloat16_emulator: Patching tmul/dpas ops is only alowed on 'cpu' device.")

    if verbose :
        print("bfloat16_emulator: Initializing bfloat16 mixed precision training..")
        if list_exempt_layers is not None:
            print("bfloat16_emulator: The following layers are excluded : {}".format(list_exempt_layers))
        if list_layers_output_fused is not None:
            print("bfloat16_emulator: The output is not converted to {} for the following layers : {}".format("bfloat16", list_layers_output_fused))

    emulator = Bfloat16Emulator(model, optimizer, device=device, verbose=verbose, tensor_stats=tensor_stats)
    emulator.enable_hw_patching(patch_ops)
    if tensor_stats :
        emulator.enable_tensor_stats()
    emulator.set_tensor_bindump_schedule(list_bindump_schedule)
    emulator.set_master_param_precision(master_params)
    emulator.set_embedding_precision(emb_precision, emb_norm)

    emulator.prepare_model(model, list_exempt_layers, list_layers_output_fused)

    if emulator.patch_ops == True and len(emulator.list_unpatched):  
        print("bfloat16_emulator: Following layers are not HW_PATCH'ed because thier dimensions do not match the hardware : {} ".format(emulator.list_unpatched))

    if emulator.verbose :
        emulator.print_config()

    return model, emulator

def prepare_model_for_inference(model, list_exempt_layers=None, device="cuda", verbose=False):
    if verbose :
        print("Initializing bfloat16 configuration for inference..")
    emulator = Bfloat16Emulator(model, None, device=device, verbose=verbose)
    emulator.emb_qconfig  = TensorQuantConfig("bfloat16", "rne")
    emulator.oact_qconfig = None
    emulator.prepare_model(model, list_exempt_layers, None)
    '''
    # preparing model for calibration
    prev_name = None
    prev_module = None
    for name,module in model.named_modules():
        if type(module) == torch.nn.BatchNorm2d \
            and prev_name in emulator.model_qconfig_dict \
            and emulator.model_qconfig_dict[prev_name].oact_qconfig is not None:
            # shifting the output quantization of BatchNorm
            assert type(prev_module) == torch.nn.Conv2d
            emulator.model_qconfig_dict[name] = ModuleQuantConfig(
                          oact_qconfig=emulator.model_qconfig_dict[prev_name].oact_qconfig,
                          patch_ops=emulator.patch_ops)
            emulator.model_qconfig_dict[prev_name].oact_qconfig = None
        # storing previous module info
        prev_name = name
        prev_module = module
    '''
    if emulator.verbose :
        emulator.print_config()
    return model, emulator
