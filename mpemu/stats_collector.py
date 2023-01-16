#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi (Intel Corporation)
#------------------------------------------------------------------------------

import torch

class TensorFullIntQuantParams(object):
    """
    min_val : float
    max_val : float
    qconfig : TensorQuantConfig
    """
    def __init__(self, min_val, max_val, qconfig):
        super(TensorFullIntQuantParams, self).__init__()
        self.qconfig = qconfig
        self.min_val, self.max_val, self.scale, self.zero_point = self._calculate_int8_qparams_base(qconfig.dtype,
                                                    qconfig.scheme, min_val, max_val)


    def quantize(self, tensor_f):
        # Clamping the values
        #tensor_f = torch.clamp(tensor_f, self.min_val, self.max_val)
        # Quantizing the tensor
        tensor_int = torch.round(tensor_f/self.scale + self.zero_point)

        min_int = -128
        max_int = 127
        if self.qconfig.dtype == "uint8":
            min_int = 0
            max_int = 255

        # Clamping the values in integer domain
        tensor_int = torch.clamp(tensor_int, min_int, max_int)

        return tensor_int

    def dequantize(self, tensor_int):
        tensor_f = (tensor_int - self.zero_point)*self.scale
        return tensor_f

    def quant_dequant(self, tensor_f):
        return self.dequantize(self.quantize(tensor_f))

    def __repr__(self):
        return "{} Quantization range [{:.4f},{:.4f}] ".format(self.qconfig, self.min_val, self.max_val)

    # Calculating quantization parameters for INT8/UINT8
    @staticmethod
    def _calculate_int8_qparams_base(dtype, scheme, min_val, max_val):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/8074779328fa471f484fb74cc6c50d95392fe2c2/torch/quantization/observer.py#L193
        """
        assert min_val <= max_val, "Minimum value {} has to be less than Maximum value {}".format(min_val,max_val)
        eps = torch.finfo(torch.float32).eps

        if dtype == "uint8":
            qmin = 0
            qmax = 255
        elif dtype == "int8":
            qmin = -128
            qmax = 127

        # Including zero in the range
        min_val,max_val = float(min_val),float(max_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)

        if min_val == max_val:
            scale = 1.0
            zero_point = 0
        else:
            if scheme == "sym_full" or scheme == "sym_channel":
                max_val = max(-min_val, max_val)
                scale = max_val / ((qmax - qmin) / 2)
                scale = max(scale, eps)
                zero_point = 0 if dtype == "int8" else 128

                min_val = -1*max_val
            elif scheme == "asym_full" or scheme == "asym_channel":
                scale = (max_val - min_val) / float(qmax - qmin)
                scale = max(scale, eps)
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)
                zero_point = int(zero_point)


        return min_val, max_val, scale, zero_point


class TensorChannelIntQuantParams(object):
    def __init__(self, min_vals, max_vals, qconfig):
        super(TensorChannelIntQuantParams, self).__init__()
        assert(len(min_vals) == len(max_vals))
        self.num_channels = len(min_vals)
        self.channel_qparams = []

        for min_val,max_val in zip(min_vals,max_vals):
            self.channel_qparams.append(TensorFullIntQuantParams(min_val, max_val, qconfig))

    def quant_dequant(self, tensor_f):
        tensor_q = torch.zeros_like(tensor_f)
        for chan_id in range(self.num_channels):
            tensor_q[chan_id] = self.channel_qparams[chan_id].quant_dequant(tensor[chan_id])
        return tensor_q


class TensorDump(torch.nn.Module):
    def __init__(self, name=""):
        self.name = name
        self.tensors = []

    def forward(self, tensor):
        self.tensors.append(tensor)

    def dump(self):
        import pickle
        import numpy as np
        pickle.dump(np.array([tensor.detach().numpy() for tensor in self.tensors]), open(self.name+".pickle","wb"))

class TensorDumpListWrapper(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""
    def __init__(self, name=""):
        super(TensorDumpListWrapper, self).__init__()
        self.initiated = False
        self.tupled_input = False
        self.tensor_dump_list = []

        self.name = name

    def forward(self, input):
        if not self.initiated:
            if type(input) == tuple:
                self.tupled_input = True
                for i in range(len(input)):
                    self.tensor_dump_list.append(TensorDump(name=self.name+"_{}".format(i)))
            else:
                self.tensor_dump_list = [TensorDump(name=self.name+"_0")]
            self.initiated = True

        if self.tupled_input:
            for i in range(len(input)):
                self.tensor_dump_list[i].forward(input[i])
        else:
            self.tensor_dump_list[0].forward(input)


    def dump(self):
        for tensor_dump in self.tensor_dump_list:
            tensor_dump.dump()


class ArchiveStats(torch.nn.Module):
    def __init__(self):
        self.tensors = []

    def forward(self, tensor):
        self.tensors.append(tensor)

class MinMaxStats(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""
    def __init__(self, archive_tensors=False):
        super(MinMaxStats, self).__init__()
        self.min_val = None
        self.max_val = None

        self.archive_tensors = archive_tensors
        self.tensors = None

    def forward(self, tensor):
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()

        if self.min_val == None:
            self.min_val = min_val
        else:
            if min_val < self.min_val:
                self.min_val = min_val

        if self.max_val == None:
            self.max_val = max_val
        else:
            if max_val > self.max_val:
                self.max_val = max_val

        if self.archive_tensors:
            if self.tensors is None:
                self.tensors = [tensor]
            else:
                self.tensors.append(tensor)

    def get_tensor_quant_params(self, ten_qconfig):
        assert(ten_qconfig.dtype in ["uint8","int8"])
        ten_qparams = TensorFullIntQuantParams(self.min_val, self.max_val, ten_qconfig)
        return ten_qparams

    def print(self, name=""):
        print("{:7.3f} {:7.3f} {}".format(self.min_val, self.max_val, name))


class RunningMinMaxStats(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""
    def __init__(self, archive_tensors=False):
        super(RunningMinMaxStats, self).__init__()

        self.min_val = None
        self.max_val = None

        self.running_min_val = None
        self.running_max_val = None

        self.running_steps = 0

        self.archive_tensors = archive_tensors
        self.tensors = None

    def forward(self, tensor):
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()

        if self.min_val == None:
            self.min_val = min_val
        else:
            if min_val < self.min_val:
                self.min_val = min_val

        if self.max_val == None:
            self.max_val = max_val
        else:
            if max_val > self.max_val:
                self.max_val = max_val

        ## Running max and running mean
        if self.running_min_val == None:
            self.running_min_val = min_val
        else:
            self.running_min_val = (self.running_min_val*self.running_steps + min_val)/(self.running_steps+1)

        if self.running_max_val == None:
            self.running_max_val = max_val
        else:
            self.running_max_val = (self.running_max_val*self.running_steps + max_val)/(self.running_steps+1)

        self.running_steps += 1

        if self.archive_tensors:
            if self.tensors is None:
                self.tensors = [tensor]
            else:
                self.tensors.append(tensor)

    def get_tensor_quant_params(self, ten_qconfig):
        assert(ten_qconfig.dtype in ["uint8","int8"])
        ten_qparams = TensorFullIntQuantParams(self.running_min_val, self.running_max_val, ten_qconfig)
        return ten_qparams

    def print(self, name=""):
        print("{:7.3f} {:7.3f} {:7.3f} {:7.3f} {}".format(self.min_val, self.max_val,
                            self.running_min_val, self.running_max_val, name))


class StatsListWrapper(torch.nn.Module):
    """docstring for MinMaxStats of a tensor"""
    def __init__(self, stats_class, archive_tensors):
        super(StatsListWrapper, self).__init__()
        self.initiated = False
        self.tupled_input = False
        self.stats_list = []

        self.stats_class = stats_class
        self.archive_tensors = archive_tensors

    def forward(self, input):
        if not self.initiated:
            if type(input) == tuple:
                self.tupled_input = True
                for i in range(len(input)):
                    self.stats_list.append(self.stats_class(archive_tensors=self.archive_tensors))
            else:
                self.stats_list = [self.stats_class(archive_tensors=self.archive_tensors)]
            self.initiated = True

        if self.tupled_input:
            for i in range(len(input)):
                self.stats_list[i].forward(input[i])
        else:
            self.stats_list[0].forward(input)

    def get_tensor_quant_params(self, ten_qconfig):
        ret_list =  [stats_obj.get_tensor_quant_params(ten_qconfig) for stats_obj in self.stats_list]

        if self.tupled_input:
            return ret_list
        else:
            return ret_list[0]

    def print(self, name=""):
        for i in range(len(self.stats_list)):
            self.stats_list[i].print(name+"[{}]".format(i))


class ChannleWiseMinMaxStats(torch.nn.Module):

    def __init__(self):
        super(ChannleWiseMinMaxStats, self).__init__()
        self.min_vals = None
        self.max_vals = None

    def forward(self, tensor):
        num_chans = tensor.shape[0]
        if self.min_vals == None or self.max_vals == None:
            self.min_vals = [None]*num_chans
            self.max_vals = [None]*num_chans

        for chan_id in range(num_chans):
            min_val = torch.min(tensor[chan_id]).item()
            max_val = torch.max(tensor[chan_id]).item()

            if self.min_vals[chan_id] == None:
                self.min_vals[chan_id] = min_val
            else:
                if self.min_vals[chan_id] < min_val:
                    self.min_vals[chan_id] = min_val

            if self.max_vals[chan_id] == None:
                self.max_vals[chan_id] = max_val
            else:
                if self.max_vals[chan_id] > max_val:
                    self.max_vals[chan_id] = max_val

    def get_tensor_quant_params(self, ten_qconfig):
        ten_qparams = TensorChannelIntQuantParams(self.min_vals, self.max_vals, ten_qconfig)
        return ten_qparams

    def print(self):
        print(self.min_vals, self.max_vals, end=' ')
