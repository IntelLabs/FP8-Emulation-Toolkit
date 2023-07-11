#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import subprocess
import itertools
import os

model_choices   = []
model_choices   += ["resnet50"]
model_choices   += ["wide_resnet50_2"]
model_choices   += ["resnext50_32x4d"]
model_choices   += ["resnext101_32x8d"]
model_choices   += ["densenet121"]
model_choices   += ["densenet201"]
model_choices   += ["mobilenet_v2"]
model_choices   += ["mobilenet_v3_small"]
model_choices   += ["mobilenet_v3_large"]
model_choices   += ["inception_v3"]
model_choices   += ["squeezenet1_1"]
model_choices   += ["efficientnet_b0"]

device_choices = ["cuda:0"]
batch_size = 64
data_type="e4m3" # Suported types : e5m2, e4m3, e3m4, hybrid  
recalibrate_bn=True#False if data_type == "e4m3" else True
num_calibration_batches = 100
finetune_lr=0.0002
finetune_epochs=2
cmodel="none"
BASE_GPU_ID = 2
NUM_GPUS = 1
print_results = False
verbose=True
data_dir="/fastdata/imagenet/"
cur_sbps = []

for exp_id, exp_config in enumerate(itertools.product(model_choices, device_choices)):
    model, device = exp_config

    # Creating output directory
    output_dir = "calib_experiments/{}-{}".format(model, device)
    dump_fp = os.path.join(output_dir,"log.txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpu_id =  BASE_GPU_ID + (exp_id % NUM_GPUS)
    cmd = ""
    if "cuda" in device:
        cmd = "CUDA_VISIBLE_DEVICES={} ".format(gpu_id)
    cmd += "python imagenet_test.py --model {} --batch-size {} --data-type {} --device {} ".format(
                model, batch_size, data_type, device)
    if cmodel != "none" and device == "cpu":
        cmd += "--patch-ops {} ".format(cmodel)
    if recalibrate_bn:
        cmd += "--recalibrate-bn --num-calibration-batches {} ".format(num_calibration_batches)
    if verbose:
        cmd += "--verbose "
    cmd += "--data-path {} --output-dir {} 2>&1 | tee {}".format(data_dir, output_dir, dump_fp)

    print(cmd)

    if print_results:
        fh = open(dump_fp)
        accuracy = float(fh.readlines()[-1].strip().split()[2].strip())
        print("{:.2f}, ".format(accuracy), end="")
        fh.close()
        continue

    p = subprocess.Popen(cmd, shell=True)
    cur_sbps.append(p)

    if exp_id%NUM_GPUS == NUM_GPUS-1:
        exit_codes = [p.wait() for p in cur_sbps]
        cur_sbps = [] # Emptying the process list
