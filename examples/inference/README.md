# FP8 Post-training Quantization

Following example demonstrates the post-training quantization flow for converting pre-trained models to use FP8 for inference.

```
# import the emulator
from mpemu import mpt_emu

...

# layers exempt from e4m3 conversion
list_exempt_layers = ["conv1","fc"]

model, emulator = mpt_emu.quantize_model (model, dtype="e4m3_rne", "None",
                               list_exempt_layers=list_exempt_layers)

# calibrate the model for a few batches of training data
evaluate(model, criterion, train_loader, device,
           num_batches=<num_calibration_batches>, train=True)

# Fuse BatchNorm layers and quantize the model
model = emulator.fuse_layers_and_quantize_model(model)

# Evaluate the quantized model
evaluate(model, criterion, test_loader, device)

```
An example demostrating post-training quantization can be found [here](./classifier/imagenet_test.py).

