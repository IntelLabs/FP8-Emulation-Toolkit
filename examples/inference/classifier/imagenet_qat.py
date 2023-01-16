#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import datetime
import os
import time
import sys
import copy
import collections

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
import torch.quantization
import utils
try:
    from apex import amp
except ImportError:
    amp = None

#from train import train_one_epoch, evaluate, load_data
from mptemu import mpt_emu

def train_one_epoch(args, model, criterion, optimizer, emulator, data_loader, 
        device, epoch, print_freq, num_batches=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    i = 0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        i += 1
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if args.data_type.lower() == "bf8":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

        if num_batches is not None and i%num_batches == 0:
            break

def evaluate(model, criterion, data_loader, device, num_batches=None, print_freq=100, train=False):
    if not train:
        model.eval()
        header = 'Test:'
    else:
        model.train()
        header = 'Train:'

    metric_logger = utils.MetricLogger(delimiter="  ")

    with torch.no_grad():
        batch_id = 0
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            if num_batches is not None and batch_id+1 == num_batches:
                break;
            else:
                batch_id += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # creating device and setting backend
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # data loading code
    print("Loading data")
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ]))
    dataset_test = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.eval_batch_size, shuffle=False,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model", args.arch)
    # Loading fp32 model
    model = torchvision.models.__dict__[args.arch](pretrained=True)
    model.to(device)

    # SGD optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)

    if args.data_type.lower() == "bf8" :
        opt_level = "O2" if args.device == "cuda" else "O0"
        loss_scale = "dynamic" if args.device == "cuda" else 1.0
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=opt_level,
                                      keep_batchnorm_fp32=True,
                                      loss_scale=loss_scale
                                      )
    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_step_size,
                                                   gamma=args.lr_gamma)
    # Loss function
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    ########################################################
    list_exempt_layers = []
    list_layers_output_fused = []
    if "resnet" in args.arch or "resnext" in args.arch:
        list_exempt_layers = ["conv1","fc"]
    elif args.arch == "vgg19_bn":
        list_exempt_layers = ["features.0", "classifier.6"]
    elif args.arch == "inception_v3":
        list_exempt_layers = ["Conv2d_1a_3x3.conv", "fc"]

    print("list of exempted layers : ", list_exempt_layers)
    model, emulator = mpt_emu.quantize_model(model, optimizer=optimizer, dtype=args.data_type.lower(), hw_patch=args.patch_ops,
                               list_exempt_layers=list_exempt_layers, list_layers_output_fused=list_layers_output_fused,
			       device=device, verbose=True)

    emulator.set_default_inference_qconfig()
    start_time = time.time()
    evaluate(model, criterion, data_loader_test, device=device)
    print('quantization-aware training : Fine-tuning the network for {} epochs'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args, model, criterion, optimizer, emulator, 
                               data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        with torch.no_grad():
            print('evaluating fine-tuned model')
            eval_model = emulator.fuse_bnlayers_and_quantize_model(model)
            #quantized_eval_model = copy.deepcopy(model)
            evaluate(eval_model, criterion, data_loader_test, device=device)
            #quantized_eval_model = emulator.fuse_bnlayers_and_quantize_model(quantized_eval_model)
            #print('evaluate quantized model')
            #evaluate(quantized_eval_model, criterion, data_loader_test, device=device)

        model.train()

        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'model_qconfig_dict' : emulator.emulator.model_qconfig_dict,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        print('saving models after {} epochs'.format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path',
                        default='/fastdata/imagenet/',
                        help='dataset')
    parser.add_argument('--arch',
                        default='mobilenet_v2',
                        help='model')
    parser.add_argument('--data-type',
                        default='bf8',
                        help='supported types : e5m2, e4m3, bf16')
    parser.add_argument('--patch-ops', default='None', 
                        help='Patch Ops to enable custom gemm implementation')
    parser.add_argument('--pruning-algo', 
                        default="None", 
                        help='Pruning method: fine-grained, unstructured, adaptive')
    parser.add_argument('--device',
                        default='cuda',
                        help='device')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='batch size for calibration/training')
    parser.add_argument('--eval-batch-size', default=256, type=int,
                        help='batch size for evaluation')
    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr',
                        default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--cache-dataset", dest="cache_dataset",\
        help="Cache the datasets for quicker initialization. \
             It also serializes the transforms",
        action="store_true",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
