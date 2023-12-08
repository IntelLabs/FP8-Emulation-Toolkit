import torch
if torch.cuda.is_available() and torch.version.hip:
    from . import fpemu as fpemu_hip
