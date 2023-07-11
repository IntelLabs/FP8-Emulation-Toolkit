```
python imagenet_main.py --arch resnet50 --enable-bf8 --master-weight-precision='fp16' --resume checkpoint.pth.tar  /fastdata/imagenet/ |& tee $LOGFILE
```
