#------------------------------------------------------------------------------ 
# Copyright (c) 2023, Intel Corporation - All rights reserved. 
# This file is part of FP8-Emulation-Toolkit
#
# SPDX-License-Identifier: BSD-3-Clause
#------------------------------------------------------------------------------
# Dharma Teja Vooturi, Naveen Mellempudi (Intel Corporation)
#------------------------------------------------------------------------------

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_test_loader(data_path, batch_size=256):
    val_dir   = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    return test_loader

arch = "resnet50"
model = torchvision.models.__dict__[arch](pretrained=True)
device = torch.device("cuda")
model.to(device)

test_loader = get_test_loader("/fastdata/imagenet")
criterion = torch.nn.CrossEntropyLoss()

from train import evaluate
def test():
    evaluate(model, criterion, test_loader, device, num_batches=2, print_freq=1)

test()

from mptemu import scale_shift
model = scale_shift.replace_batchnorms_with_scaleshifts(model) # Replacing BN with scaleshift layers
model.to(device)
test()


