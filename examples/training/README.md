# FP8 Mixed Precision Training 
Follow the examples listed here to train models using FP8 emulation toolkit. 
The toolkit supports two different training methods:  

## Direct Conversion Method  
This method uses a single FP8 format (`E5M2`) for both forward and backward computations. Operators (ex: Convolution, Linear) perform dot-product computations on input tensors expressed in `E5M2` format and accumulated into FP32 output tensors. The outut tensors are directly converted to `E5M2` before they are written to the memoryi -- gradients are scaled using automatic loss-scaling method. Full details of the training algorithm are covered in the paper [Mixed Precision Training With 8-bit Floating Point](https://arxiv.org/pdf/1905.12334.pdf).   

## Scaled-Hybrid Method
This method uses Hybrid-FP8 approach which uses `E4M3` for forward computations and `E5M2` for representing error gradients. The weight and activation tensors use `per-tensor scaling` in the forward pass to compensate for the limited dynamic range of `E4M3` format. The weight and activation scaling factors are computed every iteration at run-time which are then used to quantize the FP32 output tensor to `E4M3` format. The gradients are scaled using standard automatic loss scaling methods. This method is based on the algorithm discussed in the paper [FP8 formats for Deep Learning](https://arxiv.org/abs/2209.05433)

## Loss Scaling 
A modified NVIDIA [Apex](https://github.com/NVIDIA/apex) library is used to extend loss scaling capabilities to CPU platforms.
Follow the instructions below to patch apex library to enable CPU support.

```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ git checkout 05091d4  
$ git apply <FP8_Emulation_Toolkit/examples/training>/nvidia_apex_cpu_patch_05091d4.patch
```
Install python-only build.
```
$ pip3 install -v --no-cache-dir ./ 
```

## Usage Example

Modify the training script as follows to enable FP8 emulation. 

```
# import the emulator
from mpemu import mpt_emu

...

# layers exempt from FP8 conversion
list_exempt_layers = ["conv1","bn1","fc"]
# fused layers will be exempt from converting output tensor to FP8, the following layer will read from FP32 buffer.
list_layers_output_fused = None
# use 'direct' training method, Options : direct, hybrid
model, emulator = mpt_emu.initialize(model, optimizer, training_algo="direct", 
                              list_exempt_layers=list_exempt_layers, list_layers_output_fused=list_layers_output_fused, 
                              device="cpu", verbose=True)

...

# training loop 
for epoch in range(args.start_epoch, args.epochs):
    ...
    
    emulator.update_global_steps(epoch*len(train_loader))

    ...

    emulator.optimizer_step(optimizer)

```
