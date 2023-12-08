import torch
if torch.cuda.is_available() and torch.version.cuda:
    from . import fpemu as fpemu_cuda
