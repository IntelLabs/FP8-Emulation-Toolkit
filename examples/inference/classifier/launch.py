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

#model_choices  = ["alexnet", "vgg16_bn", "resnet50", "resnext50_32x4d", "mobilenet_v2", "mobilenet_v3_large", "shufflenet_v2_x1_0"]
#model_choices  += ["mobilenet_v3_small", "wide_resnet50_2", "resnext101_32x8d", "mnasnet1_0", "efficientnet_b7", "regnet_x_32gf"]
#model_choices   = ["alexnet", "vgg16_bn", "resnet50", "resnext50_32x4d"]
#model_choices   = ["resnet50"]
model_choices   = ["mobilenet_v2"]
#model_choices   = ["wide_resnet50_2"]
device_choices = ["cpu"]
batch_size = 64
num_calibration_batches = 500
finetune_lr=0.0002
finetune_epochs=2
data_type="e4m3"
hw_patch="none"

BASE_GPU_ID = 2
NUM_GPUS = 1
print_results = False
data_dir="/cold_storage/ml_datasets/imagenet"

cur_sbps = []
for exp_id, exp_config in enumerate(itertools.product(model_choices, device_choices)):
    model, device = exp_config

    # Creating output directory
    output_dir = "calib_experiments/{}-{}".format(model, device)
    dump_fp = os.path.join(output_dir,"log.txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gpu_id =  BASE_GPU_ID + (exp_id % NUM_GPUS)

    cmd = "CUDA_VISIBLE_DEVICES={} python imagenet_test.py --model {} \
            --batch-size {} --num-calibration-batches {} --data-type {} --patch-ops {} --device {} --data-path {} --output-dir {} 2>&1 | tee {}".format(
                gpu_id, model, batch_size, num_calibration_batches, data_type, hw_patch, device,
                data_dir, output_dir, dump_fp)
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
