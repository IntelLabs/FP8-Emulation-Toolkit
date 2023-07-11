
export MIXED_PRECISION=fp16

TRAINING_ALGO=${1:-'direct'}

python run_qa_no_trainer.py --model_name_or_path bert-large-uncased --dataset_name squad --per_device_train_batch_size 12 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --fp8_training --fp8_algo=$TRAINING_ALGO --output_dir ./bert-large-uncased-fp8 |& tee bert-large-uncased-fp8.log
