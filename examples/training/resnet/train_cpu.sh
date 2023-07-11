#!/bin/bash
export CUDA_VISIBLE_DEVICES=""

HOST_ADDR=${1:-$HOSTNAME}
BATCH=${2:-64}
TRAINING_ALGO=${3:-'direct'}
HW_PATCH=${4:-'None'}
MODEL_PREC=${5:-'fp16'}
PRUNING_ALG=${6:-'None'} # None, Unstructured, Adaptive, Auto
DATA_DIR=${7:-'/fastdata/imagenet/'}
OUTPUT_DIR=${8:-'.'}
ARCH=${9:-'resnet50'}
LR_MODE=${10:-'cosine'}

WORLD_SIZE=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
HOST_PORT=12345
LOGFILE="train.log"

CMD=" torch.distributed.launch --nnodes $WORLD_SIZE --nproc_per_node 1 --node_rank $NODE_RANK --master_addr $HOST_ADDR --master_port $HOST_PORT main_amp_cpu.py --local_rank $NODE_RANK --arch $ARCH --batch-size=$BATCH --lr-mode=$LR_MODE --fp8-training --fp8-algo=$TRAINING_ALGO --master-weight-precision=$MODEL_PREC --patch-ops=$HW_PATCH --pruning-algo $PRUNING_ALG --output-dir $OUTPUT_DIR  --resume $OUTPUT_DIR/checkpoint.pth.tar  $DATA_DIR " 

echo $CMD

python -m $CMD |& tee $LOGFILE
