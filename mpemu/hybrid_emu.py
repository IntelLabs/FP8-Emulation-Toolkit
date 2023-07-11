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

'''
Hybrid Mixed Precision Emulator
'''
class HybridEmulator(object):
    def __init__(self, model, optimizer, sparse_config=None, device="cuda", verbose=False, tensor_stats=False):
        super(HybridEmulator, self).__init__()
        self.whitelist = [torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag]
        self.whitelist += [torch.nn.LayerNorm]
        self.whitelist += [Matmul, BatchMatmul, AddMatmul]
        self.whitelist += [EltwiseAdd, EltwiseMul, EltwiseDiv]
        self.whitelist += [SparseConv2d, SparseLinear]
        self.blacklist = []
        self.list_unpatched = []
        self.list_exempt_layers = None
        self.list_layers_output_fused = None
        self.using_apex = False
        self.use_fp16_master = False
        self.use_e5m2_master = False
        self.use_e5m2_emb = False
        self.use_e4m3_emb = False
        self.use_e3m4_emb = False
        self.device = device
        self.model = model
        self.global_steps = 0
        self.emb_norm = False
        self.is_training = model.training
        self.patch_ops = False
        self.patch_impl = "NONE"
        self.patchlist = ["SIMPLE"]
        self.sparse_config = sparse_config
        # default configuration
        self.data_emulation = True
        self.mod_qconfig = None
        self.model_qconfig_dict = None #OrderedDict()
        self.emb_qconfig    = None #TensorQuantConfig("e5m2", "stochastic")
        self.wt_qconfig     = TensorQuantConfig("e4m3", "rne", "per-tensor")
        self.iact_qconfig   = TensorQuantConfig("e4m3", "rne", "per-tensor")
        self.oact_qconfig   = TensorQuantConfig("e4m3", "rne", "per-tensor")
        self.igrad_qconfig  = TensorQuantConfig("e5m2", "stochastic")
        self.ograd_qconfig  = TensorQuantConfig("e5m2", "stochastic")
        self.wtgrad_qconfig = TensorQuantConfig("e5m2", "stochastic")
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

    def _check_master_weights(self, optimizer):
        if hasattr(optimizer, "_master_params_to_model_params"):
            return True
        return False

    def update_fp16_master_params(self, optimizer) :
        if self.using_apex and self._check_master_weights(optimizer):
            from apex import amp
            if self.device == 'cuda' :
                from .pytquant.cuda import fpemu_cuda
                for param, (name, _ ) in zip(amp.master_params(optimizer), self.model.named_parameters()):
                    if self.use_e5m2_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True)
                    elif self.use_e4m3_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E4M3_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E4M3_STOCHASTIC", True)
                    elif self.use_e3m4_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E3M4_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E3M4_STOCHASTIC", True)
                    else :
                        param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)

                    if self.tensor_stats and (self.global_steps % 1000) == 0:
                        self.tb_writer.add_histogram(name, param, global_step=self.global_steps)
                        self.tb_writer.add_scalar(name+"_absmax", torch.max(torch.abs(param)), global_step=self.global_steps)
                        self.tb_writer.add_scalar(name+"_absmin", torch.min(torch.abs(param)), global_step=self.global_steps)
            else :
                from .pytquant.cpp import fpemu_cpp
                for param, (name, _) in zip(amp.master_params(optimizer), self.model.named_parameters()):
                    if self.use_e5m2_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True)
                    elif self.use_e4m3_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E4M3_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E4M3_STOCHASTIC", True)
                    elif self.use_e3m4_emb and "embeddings" in name:
                        if self.emb_norm :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E3M4_STOCHASTIC", True, self.emb_norm, param.size()[1])
                        else :
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                            param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E3M4_STOCHASTIC", True)
                    else :
                        param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)

                    if self.tensor_stats and (self.global_steps % 1000) == 0:
                        self.tb_writer.add_histogram(name, param, global_step=self.global_steps)
                        self.tb_writer.add_scalar(name+"_absmax", torch.max(torch.abs(param)), global_step=self.global_steps)
                        self.tb_writer.add_scalar(name+"_absmin", torch.min(torch.abs(param)), global_step=self.global_steps)
        else :
            print("hybrid_emulator: optimizer is not initialized with Apex, no action performed")

    def update_e5m2_master_params(self, optimizer) :
        if self.using_apex and self._check_master_weights(optimizer):
            from apex import amp
            for param in amp.master_params(optimizer):
                if self.device == 'cuda' :
                    from .pytquant.cuda import fpemu_cuda
                    param.data = fpemu_cuda.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                    param.data = fpemu_cuda.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True)
                else :
                    from .pytquant.cpp import fpemu_cpp
                    param.data = fpemu_cpp.FPEmuOp.apply(param.data, "FLOAT16_STOCHASTIC", True)
                    param.data = fpemu_cpp.FPEmuOp.apply(param.data, "E5M2_STOCHASTIC", True)
        else :
            print("hybrid_emulator: optimizer is not initialized with Apex, no action performed")

    def update_master_params(self, optimizer) :
        if self.use_fp16_master:
            self.update_fp16_master_params(optimizer)
        elif self.use_e5m2_master:
            self.update_e5m2_master_params(optimizer)

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
            if not hasattr(module, 'weight') and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = None
                self.model_qconfig_dict[name].wtgrad_qconfig = None
            """
            if type(module) not in [torch.nn.Conv2d, torch.nn.Linear, torch.nn.Embedding, torch.nn.EmbeddingBag]\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = None
                self.model_qconfig_dict[name].wtgrad_qconfig = None
            """

        for name,module in model.named_modules():
            if ((type(module) == torch.nn.Embedding) or (type(module) == torch.nn.EmbeddingBag))\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = self.emb_qconfig
                self.model_qconfig_dict[name].iact_qconfig = None
                self.model_qconfig_dict[name].igrad_qconfig = None
                self.model_qconfig_dict[name].ograd_qconfig = None
                self.model_qconfig_dict[name].oact_qconfig = None

        for name,module in model.named_modules():
            if ((type(module) == torch.nn.LayerNorm))\
                and name in self.model_qconfig_dict:
                self.model_qconfig_dict[name].wt_qconfig = None
                #self.model_qconfig_dict[name].oact_qconfig = None

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
            self.use_e5m2_master = False
            if self.verbose :
                print("hybrid_emulator: Use float16 master parameters = ", self.use_fp16_master)
        if master_params == 'e5m2' or master_params == 'BF8' :
            self.use_e5m2_master = True
            self.use_fp16_master = False
            if self.verbose :
                print("hybrid_emulator: Use e5m2 master parameters = ", self.use_e5m2_master)
 
    def set_embedding_precision(self, emb_precision, emb_norm):
        if emb_precision == 'e5m2' or emb_precision == 'BF8' :
            self.use_e5m2_emb = True
            if self.verbose :
                print("hybrid_emulator: Use e5m2 embedding table = ", self.use_e5m2_emb)
        elif emb_precision == 'e4m3' or emb_precision == 'HF8' :
            self.use_e4m3_emb = True
            if self.verbose :
                print("hybrid_emulator: Use hfloat8 embedding table = ", self.use_e4m3_emb)
        elif emb_precision == 'e4m3_2' or emb_precision == 'HF8_2' :
            self.use_e3m4_emb = True
            if self.verbose :
                print("hybrid_emulator: Use hfloat8_134 embedding table = ", self.use_e3m4_emb)
 
        if emb_norm and self.use_e5m2_emb or self.use_e4m3_emb or self.use_e3m4_emb :
            self.emb_norm = emb_norm
            if self.verbose :
                print("hybrid_emulator: Using block normalization for embeddings")

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
            print("hybrid_emulator: Enabled binary dump for iterations : {} ".format(list_bindump_schedule))
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
                print("hybrid_emulator: PyTorch Ops are monkey-patched to use {} kernels : {}".format(self.patch_impl, self.patch_ops))
            else :
                raise RuntimeError("hybrid_emulator: HW patching is not supported for {}, supported list of options : {}".format(patch_ops, self.patchlist))

    def fuse_batchnorm_with_convolution(self, model):
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        temp = []
        for name, module in model.named_children():
            if list(module.named_children()):
                self.fuse_batchnorm_with_convolution(module)

            if isinstance(module, torch.nn.BatchNorm2d):
                if isinstance(temp[-1][1], torch.nn.Conv2d):
                    setattr(model, temp[-1][0], fuse_conv_bn_eval(temp[-1][1], module))
                    setattr(model, name, torch.nn.Identity())
            else:
                temp.append((name, module))
        return model

    def set_calibration_qconfig(self):
        self.emb_qconfig    = TensorQuantConfig("e3m4", "rne", "per-channel")
        self.wt_qconfig     = TensorQuantConfig("e3m4", "rne", "per-channel")
        self.iact_qconfig   = TensorQuantConfig("e4m3", "rne", "per-tensor")
        self.oact_qconfig   = None 

    def set_default_inference_qconfig(self):
        self.emb_qconfig    = TensorQuantConfig("e3m4", "rne", "per-channel")
        self.wt_qconfig     = TensorQuantConfig("e3m4", "rne", "per-channel")
        self.iact_qconfig   = TensorQuantConfig("e4m3", "rne", "per-tensor")
        self.oact_qconfig   = None 
        self.igrad_qconfig  = None 
        self.ograd_qconfig  = None
        self.wtgrad_qconfig = None

    def fuse_layers_and_quantize_model(self, model):
        if self.is_training :
            print("Warning : emulator.is_training is set to True, returning the model unchanged")
            return model
        if self.verbose :
            print("hybrid_emulator: Fusing Batchnorm layers and replacing them with scale and shift")

        model_fused = replace_batchnorms_with_scaleshifts(model)
        self.is_training = False
        self.set_default_inference_qconfig()
        self.prepare_model(model_fused, self.list_exempt_layers, self.list_layers_output_fused)

        #reset_quantization_setup(model_fused, self.model_qconfig_dict)
        #add_quantization_hooks(model_fused, self.model_qconfig_dict)
        #quantize_model_weights(model, self.model_qconfig_dict) # added new
        #set_quantize_weights_flag(model_fused, self.model_qconfig_dict, False)
        model_fused = model_fused.to(self.device)
        if self.verbose :
            self.print_config()

        return model_fused

    def print_config(self):
        for key in self.model_qconfig_dict:
            print("{} {:40s}".format(self.model_qconfig_dict[key], key))

    def __repr__(self):
        train_infer = "inference"
        if self.is_training :
            train_infer = "training"
        return "[Configured to run {} on {}, using AMP: {}]".format(str(train_infer), self.device, str(self.using_apex))
