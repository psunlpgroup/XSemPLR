#! /bin/bash

dataset=matis
data_folder=../../dataset/
mr=sql
key=YOUR_API_KEY

for language in es de fr pt zh; do
    python run_bloom.py \
      --crosslingual \
      --hf_key=${key} \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr} \
      --num_shot=4 # MATIS input is too long, we can only use 4 shots
    python run_eval.py \
      --crosslingual \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done


