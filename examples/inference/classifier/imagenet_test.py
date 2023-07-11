#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import torch
import torchvision

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import argparse
from train import evaluate
from mpemu import mpt_emu

def get_model_exempt_layers(args, model):

    list_exempt_layers = []
    list_layers_output_fused = []
    if args.model == "alexnet":
        list_exempt_layers = ["features.0", "classifier.6"]
    elif args.model == "vgg16_bn":
        list_exempt_layers = ["features.0", "features.1", "classifier.6"]
    elif args.model == "inception_v3":
        list_exempt_layers = ["Conv2d_1a_3x3.conv", "fc"]
        #list_exempt_layers = ["fc"]
    elif args.model == "squeezenet1_1":
        list_exempt_layers = ["features.0", "classifier.1"]
        #list_exempt_layers = ["classifier.1"]
    elif "resnet" in args.model or "resnext" in args.model:
        list_exempt_layers = ["conv1","bn1","fc"]
        #list_exempt_layers = ["fc"]
    elif "densenet" in args.model:
        list_exempt_layers = ["features.conv0","features.norm0", "classifier"]
        #list_exempt_layers = ["classifier"]
    elif "mobilenet" in args.model or "efficientnet" in args.model:
        # Exempting Features[0] and classifier
        list_exempt_layers = ["features.0.0","features.0.1","classifier.1"]
        #list_exempt_layers = ["classifier.1"]
    elif args.model == "mobilenet_v3_small" or args.model == "mobilenet_v3_large":
        list_exempt_layers = ["features.0.0","features.0.1","classifier.0","classifier.3"]
        #list_exempt_layers = ["classifier.0","classifier.3"]
    elif args.model == "wide_resnet50_2":
        list_exempt_layers = ["features.0.0","features.0.1","classifier.3"]
        #list_exempt_layers = ["classifier.3"]
    elif args.model == "mnasnet1_0":
        list_exempt_layers = ["layers.0","layers.1","classifier.1"]
    elif args.model == "shufflenet_v2_x1_0":
    	list_exempt_layers = ["conv1.0","conv1.1","fc"]
    	#list_exempt_layers = ["fc"]

    prev_name = None
    prev_module = None
    for name,module in model.named_modules():
        if type(module) == torch.nn.BatchNorm2d and type(prev_module) == torch.nn.Conv2d:
            list_layers_output_fused.append(prev_name)
        if type(module) == torch.nn.Linear:
            list_layers_output_fused.append(name)

        prev_module = module
        prev_name = name

    return list_exempt_layers, list_layers_output_fused

def get_data_loaders(args):
    ### Data loader construction
    traindir = os.path.join(args.data_path, 'train')
    valdir   = os.path.join(args.data_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, test_loader

def print_bn_running_stats(model):
    rmean = None 
    rvar = None
    for name, module in model.named_children():
        if isinstance(module, torch.nn.BatchNorm2d):
            if name == "features.0.1": #"bn1":
                print("--> layer: {}, tracking running stats : {}".format( name, module.track_running_stats)) 
                #break
                rmean = module.running_mean
                rvar = module.running_var
                #print(module.running_mean)
                #print(module.running_var)
    #return  module.running_mean, module.running_var #rmean, rvar
    return  rmean, rvar
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/fastdata/imagenet/", help="Path to imagenet dataset")
    parser.add_argument("--model", default="resnet50",
                    choices=["alexnet", "vgg16_bn", "resnet18", "resnet50","resnext50_32x4d","densenet121",\
                            "densenet201","mobilenet_v2","shufflenet_v2_x1_0","mobilenet_v3_large","mobilenet_v3_small",\
                            "wide_resnet50_2","resnext101_32x8d","mnasnet1_0","efficientnet_b7","regnet_x_32gf",\
                            "inception_v3","squeezenet1_1","efficientnet_b0"],
                    help="Name of the neural network architecture")
    parser.add_argument('--data-type', default='e5m2', 
                    help='supported types : e5m2, e4m3, bf16' )
    parser.add_argument('--patch-ops', default='None', 
                    help='Patch Ops to enable custom gemm implementation')
    parser.add_argument('--device',default='cuda', help='device')
    parser.add_argument('--verbose', action="store_true", default=False, help='show debug messages')

    # Post training quantization(PTQ) related
    parser.add_argument('--recalibrate-bn', action="store_true", default=False,
                    help='Perform batchnorm recalibration')
    parser.add_argument("--num-calibration-batches", type=int, default=500,
                    help="Number of batches for BatchNorm calibration")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for train/calibrate")
    parser.add_argument('--eval-batch-size', default=32, type=int,
                    help='batch size for evaluation')
    parser.add_argument("--workers", type=int, default=16)

    # Logging and storing model
    parser.add_argument("--output-dir", default=".", help="Output directory to dump model")

    args = parser.parse_args()
    print(args)
    model_dict = {
        "alexnet" : "AlexNet_Weights.DEFAULT",
        "vgg16_bn": "VGG16_BN_Weights.DEFAULT",
        "resnet18" : "ResNet18_Weights.DEFAULT",
        "resnet50" : "ResNet50_Weights.DEFAULT",
        "resnext50_32x4d" : "ResNeXt50_32X4D_Weights.DEFAULT",
        "resnext101_32x8d" : "ResNeXt101_32X8D_Weights.DEFAULT",
        "wide_resnet50_2" : "Wide_ResNet50_2_Weights.DEFAULT",
        "densenet121" : "DenseNet121_Weights.DEFAULT",
        "densenet201" : "DenseNet201_Weights.DEFAULT",
        "mobilenet_v2" : "MobileNet_V2_Weights.DEFAULT",
        "shufflenet_v2_x1_0" : "ShuffleNet_V2_Weights.DEFAULT",
        "mobilenet_v3_large" : "MobileNet_V3_Large_Weights.DEFAULT",
        "mobilenet_v3_small" : "MobileNet_V3_Small_Weights.DEFAULT",
        "mnasnet1_0" : "MNASNet1_0_Weights.DEFAULT",
        "efficientnet_b7" : "EfficientNet_B7_Weights.DEFAULT",
        "regnet_x_32gf" : "RegNet_X_32GF_Weights.DEFAULT",
        "inception_v3" : "Inception_V3_Weights.DEFAULT",
        "squeezenet1_1" : "SqueezeNet1_1_Weights.DEFAULT",
        "efficientnet_b0" : "EfficientNet_B0_Weights.DEFAULT",
        }
    # Creating output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the model and move to GPU
    device = torch.device(args.device)
    # Data loaders and loss function
    train_loader,test_loader = get_data_loaders(args)
    criterion = torch.nn.CrossEntropyLoss()
    # Model
    model = torchvision.models.__dict__[args.model](pretrained=True)
    #model = torchvision.models.__dict__[args.model](weights=model_dict[args.model])
    model = model.to(device)
    model.eval()
    #print(model)

    print("Evaluating original {} model to establish baseline".format(args.model))
    evaluate(model, criterion, test_loader, device)

    # Create a list of exempt_layers
    list_exempt_layers, list_layers_output_fused = get_model_exempt_layers(args, model)
    print("Preparing the {} model for {} quantization".format(args.model, args.data_type.lower()))
    print("List of exempt layers : ", list_exempt_layers)
    #print(list_layers_output_fused)
    model, emulator = mpt_emu.quantize_model (model, dtype=args.data_type.lower(), calibrate=args.recalibrate_bn, hw_patch=args.patch_ops,
                               list_exempt_layers=list_exempt_layers, list_layers_output_fused=list_layers_output_fused,
    			       device=device, verbose=args.verbose)

    if args.recalibrate_bn == True:
        print("Calibrating the {} model for {} batches of training data".format(args.model, args.num_calibration_batches))
        model.train()
        evaluate(model, criterion, train_loader, device,
               num_batches=args.num_calibration_batches, train=True)
    
    model.eval()
    print("Fusing BatchNorm layers")
    model = emulator.fuse_bnlayers_and_quantize_model(model)
    print("Evaluating {} fused {} model. ".format(args.model, args.data_type.lower()))
    evaluate(model, criterion, test_loader, device)
