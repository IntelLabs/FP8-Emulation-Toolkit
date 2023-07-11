#!/bin/bash
QUANT_TYPE=${1:-'hybrid'}

# set dataset and model_path
if test -z $dataset || ! test -d $dataset ; then
  if test -d ./SQUAD1 ; then
    dataset=./SQUAD1
  else
    echo "Unable to find dataset path!!"
    echo "Download SQuAD dataset using the command ./download_squad_dataset.sh"
    exit 1
  fi
fi

if test -z $model_path || ! test -d $model_path ; then
  if test -d ./squad_finetuned_checkpoint ; then
    model_path=./squad_finetuned_checkpoint
  else
    echo "Unable to find pre-trained model path!!"
    echo "Download the pre-trained SQuAD model using the command ./download_squad_finetuned_model.sh"
    exit 1
  fi
fi

$NUMA_RAGS $GDB_ARGS python -u run_squad.py \
  --model_type bert \
  --model_name_or_path $model_path \
  --do_eval \
  --do_lower_case \
  --predict_file $dataset/dev-v1.1.json \
  --per_gpu_eval_batch_size 24 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --quant_data_type=$QUANT_TYPE \
  --output_dir /tmp/debug_squad/ \
  --no_cuda
