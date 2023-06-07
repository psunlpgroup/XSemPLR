#! /bin/bash

dataset=mnlmaps
data_folder=../../dataset/
mr=funql
key=YOUR_API_KEY
for language in en de; do
    python run_bloom.py \
      --dataset=${dataset} \
      --hf_key=${key} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr} \
      --num_shot=8
    python run_eval.py \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done


