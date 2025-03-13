#!/bin/bash
#BSUB -J gpu_test
#BSUB -n 13
#BSUB -gpu "num=1"
#BSUB -o output_%J
#BSUB -e errput_%J
#BSUB -q 4gpu

cd /share/home/px/Project/Project20_ECG_foundation_model/Model/model11_SimCLR/
module load anaconda
source activate hlt_py310_torch_new
module load cuda/11.5.0

python main_finetune.py  --model_type=cnntransf --sa_folder=cnntransf_nomix_1500_512_2500_20_75MT --downstream_tr_type=nopretrain --whe_mix_lead=nomix --logistic_batch_size=256
python main_finetune.py  --model_type=cnntransf --sa_folder=cnntransf_nomix_1500_512_2500_20_75MT --downstream_tr_type=pretrain_full --whe_mix_lead=nomix --logistic_batch_size=256
python main_finetune.py  --model_type=cnntransf --sa_folder=cnntransf_nomix_1500_512_2500_20_75MT --downstream_tr_type=pretrain_only_classifier --whe_mix_lead=nomix --logistic_batch_size=256

