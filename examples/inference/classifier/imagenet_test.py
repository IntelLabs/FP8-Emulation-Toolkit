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
    elif "resnet" in args.model or "resnext" in args.model:
        list_exempt_layers = ["conv1","bn1","fc"]
    elif "densenet" in args.model:
        list_exempt_layers = ["features.conv0","features.norm0"]
        for name,module in model.named_modules():
            if type(module) == torch.nn.BatchNorm2d:
                # First batchnorm layers in _DenseLayer module
                if name.endswith(("norm1", "norm", "norm5")):
                    list_exempt_layers.append(name)
    elif args.model == "mobilenet_v2":
        # Exempting Features[0] and classifier
        list_exempt_layers = ["features.0.0","features.0.1","classifier.1"]
        """
        # Exempting features
        start_fid = 2
        end_fid   = 18
        feature_ids = list(range(1,start_fid)) + list(range(end_fid+1,19))
        print(feature_ids)
        for feature_id in feature_ids:
            if feature_id == 1:
                list_exempt_layers += ["features.1.conv.0.0","features.1.conv.1"]
                list_exempt_layers += ["features.1.conv.0.1","features.1.conv.2"]
            elif feature_id == 18:
                list_exempt_layers += ["features.18.0"]
                list_exempt_layers += ["features.18.1"]
            else:
                # Full bottleneck layers
                list_exempt_layers += ["features.{}.conv.0.0".format(feature_id),
                                    "features.{}.conv.1.0".format(feature_id),
                                    "features.{}.conv.2".format(feature_id)]
                list_exempt_layers += ["features.{}.conv.0.1".format(feature_id),
                                    "features.{}.conv.1.1".format(feature_id),
                                    "features.{}.conv.3".format(feature_id)]
        """
    elif args.model == "mobilenet_v3_small" or args.model == "mobilenet_v3_large":
        list_exempt_layers = ["features.0.0","features.0.1","classifier.0","classifier.3"]
    elif args.model == "wide_resnet50_2":
        list_exempt_layers = ["features.0.0","features.0.1","classifier.3"]
    elif args.model == "mnasnet1_0":
        list_exempt_layers = ["layers.0","layers.1","classifier.1"]
    elif args.model == "shufflenet_v2_x1_0":
    	list_exempt_layers = ["conv1.0","conv1.1","fc"]
    	list_exempt_layers += ["stage2.0.branch1.0","stage2.0.branch1.2","stage2.0.branch2.0",
    	"stage2.0.branch2.3","stage2.0.branch2.5"]
    	list_exempt_layers += ["stage2.0.branch1.1","stage2.0.branch1.3","stage2.0.branch2.1",
    	"stage2.0.branch2.4","stage2.0.branch2.6"]

    """
    # For layers with output quantization, shift the output quantization to BN
    for name,module in model.named_modules():
        if type(module) == torch.nn.Conv2d:
            list_layers_output_fused.append(name)
    """
    prev_name = None
    prev_module = None
    for name,module in model.named_modules():
        if type(module) == torch.nn.BatchNorm2d and type(prev_module) == torch.nn.Conv2d :
            list_layers_output_fused.append(prev_name)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/fastdata/imagenet/", help="Path to imagenet dataset")
    parser.add_argument("--model", default="resnet50",
                    choices=["alexnet", "vgg16_bn", "resnet50","resnext50_32x4d","densenet121","mobilenet_v2","shufflenet_v2_x1_0","mobilenet_v3_large","mobilenet_v3_small","wide_resnet50_2","resnext101_32x8d","mnasnet1_0","efficientnet_b7","regnet_x_32gf"],
                    help="Name of the neural network architecture")
    parser.add_argument('--data-type', default='e5m2', 
                        help='supported types : e5m2, e4m3, bf16' )
    parser.add_argument('--patch-ops', default='None', 
                        help='Patch Ops to enable custom gemm implementation')
    parser.add_argument('--device',default='cuda', help='device')

    # Post training quantization(PTQ) related
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
    model = model.to(device)
    model.eval()

    print(model)
    print("Evaluating original {} model to establish baseline".format(args.model))
    evaluate(model, criterion, test_loader, device)

    # Create a list of exempt_layers
    list_exempt_layers, list_layers_output_fused = get_model_exempt_layers(args, model)
    print(list_layers_output_fused)
    print("Preparing the {} model for {} quantization".format(args.model, args.data_type.lower()))
    print("List of exempt layers : ", list_exempt_layers)
    model, emulator = mpt_emu.quantize_model (model, dtype=args.data_type.lower(), hw_patch=args.patch_ops,
                               list_exempt_layers=list_exempt_layers, list_layers_output_fused=list_layers_output_fused,
    			       device=device, verbose=False)

    print("Calibrating the {} model for {} batches of training data".format(args.model, args.num_calibration_batches))
    model.train()
    evaluate(model, criterion, train_loader, device,
               num_batches=args.num_calibration_batches, train=True)
    
    model.eval()
    print("Fusing BatchNorm layers")
    model = emulator.fuse_bnlayers_and_quantize_model(model)
    print("Evaluating {} fused {} model. ".format(args.model, args.data_type.lower()))
    print(model)
    evaluate(model, criterion, test_loader, device)
