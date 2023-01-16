#### Post Training Quantization
```
python launch.py 
```
Refer to launch.py for more details.

#### QAT(Quantization Aware Training)
```
python imagenet_qat.py  \
    --data-path <path_to_imagenet_data> \
    --arch=mobilenet_v2 \
    --lr=0.001 \
    --epochs=15 \
    --lr-step-size=5 \
    --lr-gamma=0.1 \
    --qdtype bfloat8 --qscheme rne --qlevel 3
```
