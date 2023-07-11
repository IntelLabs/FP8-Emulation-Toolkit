
Prerequsite:
-----------

Install squad task specific requirements (one time):
$pip install -r requirements.txt

Download SQUAD dataset:
----------------------
$bash download_squad_dataset.sh

Download Squad fine-tuned model for inference:
---------------------------------------------
$bash download_squad_fine_tuned_model.sh

To run squad baseline inference task:
$bash cmd_infer.sh 

To run squad inference in BF8:
$bash cmd_infer.sh --use_pcl --pcl_bf8 --unpad

