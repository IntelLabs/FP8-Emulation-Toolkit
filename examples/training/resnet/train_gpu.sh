#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

HOST_ADDR=${1:-$HOSTNAME}
BATCH=${2:-256}
TRAINING_ALGO=${3:-'direct'}
HW_PATCH=${4:-'None'}
MODEL_PREC=${5:-'fp16'}
PRUNING_ALG=${6:-'None'} # None, Unstructured, Adaptive, Auto
DATA_DIR=${7:-'/fastdata/imagenet/'}
OUTPUT_DIR=${8:-'.'}
ARCH=${9:-'resnet50'}
LR_MODE=${10:-'cosine'}

WORLD_SIZE=${SLURM_JOB_NUM_NODES:-4}
NODE_RANK=${SLURM_NODEID:-0}
HOST_PORT=12345
LOGFILE="train.log"

#CMD="torchrun --nnodes=1:4 --nproc_per_node 4 --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR main_amp.py --arch $ARCH --batch-size=$BATCH --lr-mode=$LR_MODE --fp8-training --fp8-algo=$TRAINING_ALGO --master-weight-precision=$MODEL_PREC --pruning-algo $PRUNING_ALG --momentum 0.875 --lr 0.256 --weight-decay 3.0517578125e-05  --output-dir $OUTPUT_DIR  --resume $OUTPUT_DIR/checkpoint.pth.tar  $DATA_DIR " 
CMD=" torch.distributed.launch --nproc_per_node 4 --master_addr $HOST_ADDR --master_port $HOST_PORT main_amp.py --arch $ARCH --batch-size=$BATCH --lr-mode=$LR_MODE --fp8-training --fp8-algo=$TRAINING_ALGO --master-weight-precision=$MODEL_PREC --pruning-algo $PRUNING_ALG --momentum 0.875 --lr 0.256 --weight-decay 3.0517578125e-05  --output-dir $OUTPUT_DIR  --resume $OUTPUT_DIR/checkpoint.pth.tar  $DATA_DIR " 

echo $CMD

python -m $CMD |& tee $LOGFILE
