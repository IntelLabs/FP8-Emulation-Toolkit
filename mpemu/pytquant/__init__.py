import warnings
import torch 

from . import cpp
from .cpp import fpemu_cpp

if torch.cuda.is_available() and torch.version.cuda:
    from . import cuda
    from .cuda import fpemu_cuda as fpemu_cuda

if torch.cuda.is_available() and torch.version.hip:
    from . import hip
    from .hip import fpemu_hip as fpemu_cuda
