#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

from collections import OrderedDict
import copy

import torch
import numpy as np
from .stats_collector import StatsListWrapper, MinMaxStats, RunningMinMaxStats
from .stats_collector import TensorFullIntQuantParams,TensorChannelIntQuantParams
from .stats_collector import TensorDumpListWrapper

"""
 Quantization configuration for a tensor
"""
class TensorQuantConfig(object):
    def __init__(self, dtype=None, scheme=None, scaling="None", group_size=1):
        super(TensorQuantConfig, self).__init__()
        self.dtype = dtype
        self.scheme = scheme
        #self.scaling = True if scaling in ["per-tensor", "per-channel"] else False
        #self.per_channel = True if scaling == "per-channel" else False
        self.scaling = True if "tensor" in scaling.split("-") else False
        self.per_channel = True if "channel" in scaling.split("-") else False
        #self.scaling_method = ("mean" if "mean" in scaling.split("-") else "max" if "per" in scaling.split("-") or self.scaling is True else "none")
        self.scaling_method = ("mean" if "mean" in scaling.split("-") else "max")
        #if self.dtype == "int8" or self.dtype == "int4":
        if "int" in self.dtype:
            self.scaling_method = "none"

        self.fine_grained = True if scaling == "fine-grained" else False  
        self.group_size = group_size if self.fine_grained is True else 1

        self.is_enabled = True

        self.dtypes_all = ["e5m2","e4m3","e3m4","fp4","bfloat16","float16"]
        self.e5m2_modes = ["rtz", "stochastic", "rne", "rnaz", "rntz", "rpinf", "rninf"]
        self.e5m2_modes += ["daz_stochastic", "daz_rne", "daz_rnaz", "daz_rntz"]
        self.e4m3_modes = ["rne", "stochastic"]
        self.e4m3_ieee_modes = ["ieee_rne", "ieee_stochastic"]
        self.e4m3_all_modes = self.e4m3_modes + self.e4m3_ieee_modes
        self.e3m4_modes = ["rne", "stochastic"]
        self.fp4_modes = ["nearest"]
        self.bfloat16_modes = ["rne", "stochastic"]
        self.float16_modesp = ["rne", "daz_rne"]

        self.check_validity(dtype, scheme)

    def check_validity(self, dtype, scheme):
        """
        E5M2_RTZ 
        E5M2_STOCHASTIC 
        E5M2_RNE 
        E5M2_RNAZ 
        E5M2_RNTZ 
        E5M2_RPINF 
        E5M2_RNINF 
        E5M2_DAZ_STOCHASTIC 
        E5M2_DAZ_RNE 
        E5M2_DAZ_RNAZ
        E5M2_DAZ_RNTZ
        E4M3_RNE
        E4M3_STOCHASTIC
        E4M3_IEEE_RNE
        E4M3_IEEE_STOCHASTIC
        E3M4_RNE
        E3M4_STOCHASTIC
        # 16-bit types
        BFLOAT16_STOCHASTIC
        BFLOAT16_RNE
        FLOAT16_RNE
        FLOAT16_STOCHASTIC
        FLOAT16_DAZ_RNE
        """
        # integers of all sizes are acceptable
        if "int" in dtype: 
            return 

        assert dtype in self.dtypes_all, print("Invalid data type, list of supported types :".format(self.dtypes_all))

        if dtype in ["e5m2"]:
            assert scheme in self.e5m2_modes, print("scheme {} is not in ".format(scheme), self.e5m2_modes)
        elif dtype in ["e4m3"]:
            assert scheme in self.e4m3_all_modes, print("scheme {} is not in ".format(scheme), self.e4m3_all_modes)
        elif dtype in ["e3m4"]:
            assert scheme in self.e3m4_modes, print("scheme {} is not in ".format(scheme), self.e3m4_modes)
        elif dtype in ["fp4"]:
            assert scheme in self.fp4_modes, print("scheme {} is not in ".format(scheme), self.fp4_modes)
        elif dtype in ["bfloat16"]:
            assert scheme in self.bfloat16_modes, print("scheme {} is not in ".format(scheme), self.bfloat16_modes)
        elif dtype in ["float16"]:
            assert scheme in self.float16_modes, print("scheme {} is not in ".format(scheme), self.float16_modes)

    def get_flt_max(self):
        if self.dtype in ["e5m2"]:
            if self.scheme in self.e5m2_modes:
                return float(57344.0)
        elif self.dtype in ["e4m3"]:
            if self.scheme in self.e4m3_modes:
                return float(448.0)
            elif self.scheme in self.e4m3_ieee_modes:
                return float(240.0)
        elif self.dtype in ["e3m4"]:
            if self.scheme in self.e3m4_modes:
                return float(30.0)
        elif self.dtype in ["fp4"]:
            if self.scheme in self.fp4_modes:
                return float(1.0)

    def get_flt_min(self):
        if self.dtype in ["e5m2"]:
            if self.scheme in self.e5m2_modes:
                return float(1.5258789E-05)
        elif self.dtype in ["e4m3"]:
            if self.scheme in self.e4m3_all_modes:
                return float(1.9531250E-03)
        elif self.dtype in ["e3m4"]:
            if self.scheme in self.e3m4_modes:
                return float(1.5625000E-02)
        elif self.dtype in ["fp4"]:
            if self.scheme in self.fp4_modes:
                return float(0.000244140625)


    def __repr__(self):
        return "[{}, scale: {}, method: {}]".format(self.dtype+"_"+self.scheme, ("per-channel" if self.per_channel is True else "per-tensor" if self.scaling is True else "fine-grained" if self.fine_grained is True else "None"), self.scaling_method)

"""
  Quantization configuration for a module
"""
class ModuleQuantConfig(object):
    def __init__(self, wt_qconfig=None, iact_qconfig=None, oact_qconfig=None,
                wtgrad_qconfig=None, igrad_qconfig=None, ograd_qconfig=None, patch_ops=False):
        super(ModuleQuantConfig, self).__init__()
        self.wt_qconfig     = wt_qconfig
        self.iact_qconfig   = iact_qconfig
        self.oact_qconfig   = oact_qconfig
        self.wtgrad_qconfig = wtgrad_qconfig
        self.igrad_qconfig  = igrad_qconfig
        self.ograd_qconfig  = ograd_qconfig

        self.device = 'cuda'
        self.patch_ops = patch_ops
        self.patch_impl = "None"
        self.sparse_config = None
        self.global_step = -1
        self.bin_dump = False
        self.tensor_stats = False
        self.tb_writer = None
        # master copy of the weight, ueed for bfloat16 emulation
        self.weight_master = None
    
    def register_output_grad_hook(self, output):
        # output graident hook
        def quantize_output_grad(grad):
            if self.sparse_config is not None :
                if self.sparse_config.outgrad :
                    grad.data = self.sparse_config.sparsify_outgrad_tensor (grad.data)

            if self.ograd_qconfig is not None and self.ograd_qconfig.is_enabled: 
                grad = quantize_tensor(grad, self.ograd_qconfig, None, inplace=False)
            return grad
        # registering hook for output graidents
        if output.requires_grad :
            output.register_hook(quantize_output_grad)

    def register_weight_grad_hook(self, module):
        # weight graident hook
        def quantize_weight_grad(grad):
            # sparsify weight gradients 
            if self.sparse_config is not None :
                if self.sparse_config.wtgrad :
                    grad.data = self.sparse_config.sparsify_wtgrad_tensor (grad.data)

            if self.wtgrad_qconfig is not None and self.wtgrad_qconfig.is_enabled: 
                grad.data = quantize_tensor(grad.data, self.wtgrad_qconfig, None, inplace=True)
            return grad

        # registering hook for weight graidents
        if type(module) in [torch.nn.Conv2d, torch.nn.Linear]:
            if module.weight is not None :
                if module.weight.requires_grad :
                    module.weight.register_hook(quantize_weight_grad)

    def __repr__(self):
        if self.igrad_qconfig is None:
            #this is inference
            return "[weights: {}, inputs: {}, output: {}]".format(self.wt_qconfig, self.iact_qconfig, self.oact_qconfig)
        else :
            #this is training
            return "[weights: {}, inputs: {}, output: {}, weight_grad: {}, input_grad: {}, output_grad: {}]".format(
            self.wt_qconfig, self.iact_qconfig, self.oact_qconfig, self.wtgrad_qconfig, self.igrad_qconfig, self.ograd_qconfig)

"""
 Quantization parameters for quantization methods used in a module.
"""

class ModuleQuantParams(object):
    def __init__(self, wt_qparams=None, iact_qparams=None, oact_qparams=None,
                    wtgrad_qparams=None, igrad_qparams=None, ograd_qparams=None):
        super(ModuleQuantParams, self).__init__()
        self.wt_qparams     = wt_qparams
        self.iact_qparams   = iact_qparams
        self.oact_qparams   = oact_qparams

        self.wtgrad_qparams = wtgrad_qparams
        self.igrad_qparams  = igrad_qparams
        self.ograd_qparams  = ograd_qparams

def quantize_to_integer(tensor, mode, inplace=False):
    # compute tensor min and max values
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # int8 quantization range 

    nbits = int(mode.split("INT")[1])-1
    q_min = -1*2**nbits
    q_max = (2**nbits)-1

    """
    q_min = -128
    q_max = 127
    if mode == "INT4":
        q_min = -8
        q_max = 7
    """
    # compute scale and zero_point 
    scale = (max_val - min_val) / (q_max - q_min)
    zero_point = q_min - (min_val / scale)
    # Quantize the input tensor using int8 representation
    qtensor = torch.round((tensor / scale) + zero_point)
    # Clamp the values to the int8 range
    qtensor = torch.clamp(qtensor, q_min, q_max)
    # Dequantize the tensor
    dqtensor = scale * (qtensor - zero_point)

    if inplace is True:
        tensor.data.copy_(dqtensor)
        return tensor
    
    return dqtensor

def fpemu_device_fn(tensor, mode, inplace=True, scale=1.0):

    #if "INT8" in mode or "INT4" in mode:
    if "INT" in mode:
        return quantize_to_integer(tensor, mode.split("_")[0], inplace=inplace)

    if tensor.is_cuda :
        from .pytquant import fpemu_cuda
        tensor_q = fpemu_cuda.FPEmuOp.apply(tensor, mode, inplace, scale)
    else :
        from .pytquant import fpemu_cpp
        tensor_q = fpemu_cpp.FPEmuOp.apply(tensor, mode, inplace, scale)
    return tensor_q

# Single entry function for quantizing tensor.
def quantize_tensor(tensor, qtconfig, qtparams, inplace=False ):
    qtconfig.check_validity(qtconfig.dtype, qtconfig.scheme)

    mode = qtconfig.dtype.upper()
    if qtconfig.scheme is not None:
        mode += "_"+qtconfig.scheme.upper()
    # grda number of channels, assume NCHW or KCRS layout
    if inplace is False:
        tensor_q = torch.zeros_like(tensor)
    import statistics
    if qtconfig.scaling is True:
        scale = 1.0
        if qtconfig.scaling_method == "mean":
            mean = torch.mean(abs(torch.flatten(tensor.detach())))
            mean = abs(mean) if abs(mean) > 1e-5 else qtconfig.get_flt_min() 
            if abs(mean) > 0.0:
                scale = qtconfig.get_flt_min()/ abs(mean)
            scale = 1.0 if scale < 1.0 else scale
        elif qtconfig.scaling_method == "max":
            vmax = torch.max(abs(torch.flatten(tensor.detach())))
            scale = qtconfig.get_flt_max()/vmax
            scale = 6.55e+04 if scale > 3.275e+04 else scale
        
        return fpemu_device_fn(tensor, mode, inplace=inplace, scale=scale)

    elif qtconfig.per_channel is True:
        channels = tensor.shape[0]
        for c in range(channels):
            sub_tensor = tensor.select(0, c).detach()
            scale = 1.0
            if qtconfig.scaling_method == "mean":
                #mean = abs(torch.mean(torch.flatten(sub_tensor)))
                #mean = torch.mean(abs(torch.flatten(sub_tensor)))
                mean = abs(torch.mode(torch.flatten(sub_tensor)).values.data)#, dim=-1, keepdim=False))
                mean = abs(mean) if abs(mean) > 1e-6 else qtconfig.get_flt_min() 
                if abs(mean) > 0.0:
                    scale = qtconfig.get_flt_min()/ abs(mean)
                scale = 1.0 if scale < 1.0 else scale
            elif qtconfig.scaling_method == "max":
                vmax = torch.max(abs(torch.flatten(sub_tensor)))
                scale = qtconfig.get_flt_max()/vmax
                scale = 6.55e+04 if scale > 3.275e+04 else scale

            sub_tensor = fpemu_device_fn(sub_tensor, mode, inplace=False, scale=scale)
            if inplace is False:
                tensor_q.select(0, c).data.copy_(sub_tensor)
            else:
                tensor.select(0, c).data.copy_(sub_tensor)

    elif qtconfig.fine_grained is True:
        out_channels = tensor.shape[0]
        for k in range(out_channels):
            crs_tensor = tensor.select(0, k).detach()
            in_channels = crs_tensor.shape[0]
            chunks = max(1, int(in_channels/qtconfig.group_size))
            sub_tensors = crs_tensor.chunk(chunks, 0)
            for sub_tensor in sub_tensors:
                scale = 1.0
                if qtconfig.scaling_method == "mean":
                    mean = abs(torch.mode(torch.flatten(sub_tensor)).values.data)
                    mean = abs(mean) if abs(mean) > 1e-6 else qtconfig.get_flt_min() 
                    if abs(mean) > 0.0:
                        scale = qtconfig.get_flt_min()/ abs(mean)
                    scale = 1.0 if scale < 1.0 else scale
                elif qtconfig.scaling_method == "max":
                    vmax = torch.max(abs(torch.flatten(sub_tensor)))
                    scale = qtconfig.get_flt_max()/vmax
             
                sub_tensor = fpemu_device_fn(sub_tensor, mode, inplace=True, scale=scale)

            if inplace is False:
                tensor_q.select(0, k).data.copy_(crs_tensor)
            else:
                tensor.select(0, k).data.copy_(crs_tensor)

    else:
        return fpemu_device_fn(tensor, mode, inplace=inplace)

    if inplace is False:
        return tensor_q

    return tensor

"""
  Filters all the modules of type in filter_module_types
  and generates configuration dictionary.
"""
def get_or_update_model_quant_config_dict(model, filter_module_types, mod_qconfig, prefix="",
                                model_qconfig_dict=None, override=False, exempt_modules=[]):
    if model_qconfig_dict is None:
        model_qconfig_dict = OrderedDict()

    for name,module in model.named_modules():
        fname = prefix+name
        if type(module) in filter_module_types and fname not in exempt_modules:
            if fname not in model_qconfig_dict or override:
                model_qconfig_dict[fname] = copy.deepcopy(mod_qconfig)
        elif type(module) not in filter_module_types :
            if fname in model_qconfig_dict :
                model_qconfig_dict.pop(fname)

    return model_qconfig_dict

def stats_forward_hook(self, input, output):
    self.weight_pre_process(self.weight.data)
    self.activation_pre_process(input)
    self.activation_post_process(output)

def add_stats_collector_hooks(model, model_qconfig_dict,
                            stats_class_map=None, stats_class=RunningMinMaxStats,
                            archive_tensors=False):
    hook_handles = []
    for name, module in model.named_modules():
        if name in model_qconfig_dict:
            ch_stats_class = stats_class if stats_class_map is None else stats_class_map[name]
            module.add_module("weight_pre_process", StatsListWrapper(ch_stats_class, archive_tensors))
            module.add_module("activation_pre_process", StatsListWrapper(ch_stats_class, archive_tensors))
            module.add_module("activation_post_process", StatsListWrapper(ch_stats_class, archive_tensors))
            handle = module.register_forward_hook(stats_forward_hook)
            hook_handles.append(handle)

    return hook_handles

class TensorDumpHelper():
    """docstring for TensorDumpHelper"""
    def __init__(self, model, layer_names):
        super(TensorDumpHelper, self).__init__()
        self.model = model
        self.layer_names = layer_names

        # Adding hooks to the model
        hook_handles = []
        for name, module in model.named_modules():
            if name in layer_names:
                module.add_module("weight_pre_process", TensorDumpListWrapper(name=name+".weight"))
                module.add_module("activation_pre_process", TensorDumpListWrapper(name=name+".input"))
                module.add_module("activation_post_process", TensorDumpListWrapper(name=name+".output"))
                handle = module.register_forward_hook(stats_forward_hook)
                hook_handles.append(handle)

        self.hook_handles = hook_handles

    def dump_tensors(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                module.weight_pre_process.dump()
                module.activation_pre_process.dump()
                module.activation_post_process.dump()


def bindump_tensor(t, step, module_name, prefix):
    filename = str(step)+"_"+module_name+"_"+prefix+"_"
    for d in range(t.dim()):
        filename = filename+str(t.size()[d])+"_"
    filename = filename+str(t.dtype).split(".")[1]+".bin"
    print(filename)
    if t.dtype == torch.float32:
        bits = np.asarray(t.cpu().detach().numpy(), dtype=np.float32).view(np.int32)
    elif t.dtype == torch.float16:
        bits = np.asarray(t.cpu().detach().numpy(), dtype=np.float16).view(np.int16)
    else:
        return
    bits.tofile(filename)
    return

def calculate_int8_qparams(tensor, qconfig):
    assert(qconfig.dtype in ["int8","uint8"])
    #assert(qconfig.scheme in ["sym_full","asym_full","sym_channel","asym_channel"])
    if qconfig.scheme in ["sym_full", "asym_full"]:
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()
        ten_qparams = TensorFullIntQuantParams(min_val, max_val, qconfig)
        return ten_qparams
    elif qconfig.scheme in ["sym_channel","asym_channel"]:
        num_chans = tensor.shape[0]
        min_vals = []
        max_vals = []
        for chan_id in range(num_chans):
            min_vals.append(torch.min(tensor[chan_id]).item())
            max_vals.append(torch.max(tensor[chan_id]).item())
        ten_qparams = TensorChannelIntQuantParams(min_vals, max_vals, qconfig)
        return ten_qparams

# Single entry functions for getting quantization parameters
def get_quantization_parameters(tensor, qtconfig):
    #if qtconfig.dtype in ["uint8","int8"]:
    #    return calculate_int8_qparams(tensor, qtconfig)
    if qtconfig.dtype in ["e5m2","e4m3","e3m4","fp4","bfloat16","float16","int8","int4"]:
        # Stateless quantization schemes have no parameters
        return None
    else:
        print("dtype %s is invalid".format(qconfig.dtype))
        exit(-1)

def quantize_weights_of_a_module(module, wt_qconfig, name=""):
    # TEMPORARY
    wt_qparams = None #get_quantization_parameters(module.weight.data, wt_qconfig)
    module.weight.data.copy_(quantize_tensor(module.weight.data, wt_qconfig, wt_qparams))
    """  
    if hasattr(module, 'bias'):
        if module.bias is not None: 
            bias_qparams = get_quantization_parameters(module.bias.data, wt_qconfig)
            module.bias.data.copy_(quantize_tensor(module.bias.data, wt_qconfig, wt_qparams))
    """
    return

def quantize_model_weights(model, model_qconfig_dict):
    for name,module in model.named_modules():
        # Quantize weights only if they are asked to
        if name in model_qconfig_dict and model_qconfig_dict[name].wt_qconfig is not None:
            wt_qconfig = model_qconfig_dict[name].wt_qconfig
            quantize_weights_of_a_module(module, wt_qconfig, name=name)


def quantize_module_weights_and_inputs(self, input):
    # Patching ops if set
    if self.qconfig.patch_ops and len([_ for _ in self.parameters()]) != 0 :
        if next(self.parameters()).is_cuda is True :
            pass
        else:
            try:
                from .cmodel import simple
            except ImportError:
                raise ImportError("Monkey patching with custom OPs is not supported at this time.")

            # Backup op functions
            self.torch_addmm = torch.addmm
            self.torch_matmul = torch.matmul
            self.torch_mm = torch.mm
            self.torch_Tensor_addmm = torch.Tensor.addmm
            self.torch_Tensor_matmul = torch.Tensor.matmul
            self.torch_Tensor_mm = torch.Tensor.mm
            self.torch_functional_conv2d = torch.nn.functional.conv2d
            # Replace op functions
            hw_impl = None
            if self.qconfig.patch_impl == "SIMPLE":
                from .cmodel import simple as hw_impl
 
            if hw_impl is not None :
                torch.addmm = hw_impl.addmm
                torch.matmul = hw_impl.matmul
                torch.mm = hw_impl.mm
                torch.Tensor.addmm = hw_impl.addmm
                torch.Tensor.matmul = hw_impl.matmul
                torch.Tensor.mm = hw_impl.mm
                torch.nn.functional.conv2d = hw_impl.conv2d

    if self.qconfig.wt_qconfig is not None and (self.qconfig.wt_qconfig.dtype in ["bfloat16"] or self.qconfig.device == 'cpu'):
        self.qconfig.weight_master = copy.deepcopy(self.weight.data)

    # Sparsify weights inplace
    if self.qconfig.sparse_config is not None :
        if self.qconfig.sparse_config.weight and type(self) in [torch.nn.Conv2d, torch.nn.Linear]:
            self.weight.data = self.qconfig.sparse_config.sparsify_weight_tensor(self.weight.data)

    # Quantize weights inplace
    if self.qconfig.wt_qconfig is not None and self.qconfig.wt_qconfig.is_enabled:
        self.weight.data = quantize_tensor(self.weight.data, self.qconfig.wt_qconfig,
			self.qparams.wt_qparams, inplace=True)

    # registering hook for weight graidents
    #self.qconfig.register_weight_grad_hook(self)

    # Quantizing inputs
    if self.qconfig.iact_qconfig is not None and self.qconfig.iact_qconfig.is_enabled:
        input_q = []
        for i in range(len(input)):
            iact_qparams = None if self.qparams.iact_qparams is None else self.qparams.iact_qparams[i]
            tensor_q = quantize_tensor(input[i], self.qconfig.iact_qconfig, iact_qparams, inplace=False)
            input_q.append(tensor_q)
        input = tuple(input_q)

    if self.qconfig.bin_dump :
        if self.qconfig.wt_qconfig is not None and self.qconfig.wt_qconfig.is_enabled:
            bindump_tensor(self.weight, self.qconfig.global_step, self.name, "weight")
        if self.qconfig.iact_qconfig is not None and self.qconfig.iact_qconfig.is_enabled:
            for i in range(len(input)):
                bindump_tensor(input[i], self.qconfig.global_step, self.name, "input"+"_"+str(i))

    if self.qconfig.tensor_stats :
        for i in range(len(input)):
            self.qconfig.tb_writer.add_histogram(self.name+"_input"+"_"+str(i), input[i], global_step=self.qconfig.global_step)
            self.qconfig.tb_writer.add_scalars(self.name+"_input"+"_"+str(i), {'abs_max':torch.max(torch.abs(input[i])),
                                'abs_min':torch.min(torch.abs(input[i][input[i]!=0.0]))}, global_step=self.qconfig.global_step)

    return input

def quantize_module_gradient(self, grad_input, grad_output) :
    if self.qconfig.bin_dump :
        g = 0
        for in_grad, out_grad in zip(grad_input, grad_output):
            bindump_tensor(in_grad, self.qconfig.global_step, self.name, "input_grad"+"_"+str(g))
            bindump_tensor(out_grad, self.qconfig.global_step, self.name, "output_grad"+"_"+str(g))
            g = g+1
        if self.weight.grad is not None :
            bindump_tensor(self.weight.grad, self.qconfig.global_step, self.name, "weight_grad")

    if self.qconfig.sparse_config is not None :
        if self.qconfig.sparse_config.ingrad :
            grad_input = tuple([self.qconfig.sparse_config.sparsify_ingrad_tensor(grad) for grad in grad_input])

    if self.qconfig.igrad_qconfig is not None and self.qconfig.igrad_qconfig.is_enabled:
        grad_input = tuple([quantize_tensor(grad, self.qconfig.igrad_qconfig, self.qparams.igrad_qparams, inplace=False)\
                            for grad in grad_input])

    if self.qconfig.tensor_stats :
        g = 0
        for in_grad, out_grad in zip(grad_input, grad_output):
            self.qconfig.tb_writer.add_histogram(self.name+"_input_grad"+"_"+str(g), in_grad, global_step=self.qconfig.global_step)
            self.qconfig.tb_writer.add_histogram(self.name+"_output_grad"+"_"+str(g), out_grad, global_step=self.qconfig.global_step)
            self.qconfig.tb_writer.add_scalars(self.name+"_input_grad"+"_"+str(g), {'abs_max':torch.max(torch.abs(in_grad)),
                                'abs_min':torch.min(torch.abs(in_grad[in_grad!=0.0]))}, global_step=self.qconfig.global_step)
            self.qconfig.tb_writer.add_scalars(self.name+"_output_grad"+"_"+str(g), {'abs_max':torch.max(torch.abs(out_grad)),
                                'abs_min':torch.min(torch.abs(out_grad[out_grad!=0.0]))}, global_step=self.qconfig.global_step)
            g = g+1

    # copying the master weights back to original weight tensors
    if self.qconfig.wt_qconfig is not None and (self.qconfig.wt_qconfig.dtype in ["bfloat16"] or self.qconfig.device == 'cpu'):
        self.weight.data = copy.deepcopy(self.qconfig.weight_master )

    return grad_input

def quantize_module_output(self, input, output):

    if self.qconfig.patch_ops and len([_ for _ in self.parameters()]) != 0 :
        if next(self.parameters()).is_cuda is True :
            pass
        else:
            # Restore op functions
            if self.qconfig.patch_impl != "None":
                torch.addmm = self.torch_addmm
                torch.matmul = self.torch_matmul
                torch.mm = self.torch_mm
                torch.Tensor.addmm = self.torch_Tensor_addmm
                torch.Tensor.matmul = self.torch_Tensor_matmul
                torch.Tensor.mm = self.torch_Tensor_mm
                torch.nn.functional.conv2d = self.torch_functional_conv2d
    # registering hook for output graidents
    self.qconfig.register_output_grad_hook(output)

    if self.qconfig.bin_dump:
        bindump_tensor(output, self.qconfig.global_step, self.name, "output")

    # Quantizing outputs
    if self.qconfig.oact_qconfig is not None and self.qconfig.oact_qconfig.is_enabled :
        output = quantize_tensor(output, self.qconfig.oact_qconfig, self.qparams.oact_qparams, inplace=True)

    if self.qconfig.tensor_stats :
        self.qconfig.tb_writer.add_histogram(self.name+"_output", output, global_step=self.qconfig.global_step)
        self.qconfig.tb_writer.add_scalars(self.name+"_output", {'abs_max':torch.max(torch.abs(output)),
                    'abs_min':torch.min(torch.abs(output[output!=0.0]))}, global_step=self.qconfig.global_step)

    return output

def add_quantization_hooks(model, model_qconfig_dict, is_training=False):
    hook_handles = []
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            handle_pf = module.register_forward_pre_hook(quantize_module_weights_and_inputs)
            hook_handles.append(handle_pf)
            handle_f = module.register_forward_hook(quantize_module_output)
            hook_handles.append(handle_f)
            if is_training:
                if module.qconfig.igrad_qconfig is not None :
                    handle_b = module.register_full_backward_hook(quantize_module_gradient)
                    hook_handles.append(handle_b)
                # registering hook for weight graidents
                module.qconfig.register_weight_grad_hook(module)

    return hook_handles


def reset_quantization_setup(model, model_qconfig_dict):
    # Add extra attributes to chosen modules
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            module.name = name
            module.qconfig = copy.deepcopy(model_qconfig_dict[name])
            module.qparams = ModuleQuantParams()

def set_quantize_weights_flag(model, model_qconfig_dict, set_value):
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            if module.qconfig.wt_qconfig is not None:
                module.qconfig.wt_qconfig.is_enabled = set_value

def set_quantize_outputs_flag(model, model_qconfig_dict, set_value):
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            if module.qconfig.oact_qconfig is not None:
                module.qconfig.oact_qconfig.is_enabled = set_value

def set_quantize_inputs_flag(model, model_qconfig_dict, set_value):
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            if module.qconfig.iact_qconfig is not None:
                module.qconfig.iact_qconfig.is_enabled = set_value

def set_qparams_for_modules_using_stats_info(model, model_qconfig_dict):
    for name,module in model.named_modules():
        if name in model_qconfig_dict:
            wt_qparams = None
            wt_qconfig = model_qconfig_dict[name].wt_qconfig
            if wt_qconfig != None and wt_qconfig.dtype in ["uint8","int8"]:
                wt_qparams = module.weight_pre_process.get_tensor_quant_params(wt_qconfig)

            iact_qparams = None
            iact_qconfig = model_qconfig_dict[name].iact_qconfig
            if iact_qconfig != None and iact_qconfig.dtype in ["uint8","int8"]:
                iact_qparams = module.activation_pre_process.get_tensor_quant_params(iact_qconfig)

            oact_qparams = None
            oact_qconfig = model_qconfig_dict[name].oact_qconfig
            if oact_qconfig != None and oact_qconfig.dtype in ["uint8","int8"]:
                oact_qparams = module.activation_post_process.get_tensor_quant_params(oact_qconfig)

            # Setting module quantization parameters
            module.qparams = ModuleQuantParams(wt_qparams=wt_qparams,
                                               iact_qparams=iact_qparams,
                                               oact_qparams=oact_qparams)

