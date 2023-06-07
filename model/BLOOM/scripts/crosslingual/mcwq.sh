#! /bin/bash

dataset=mcwq
data_folder=../../dataset/
mr=sparql
key=YOUR_API_KEY

for language in he kn zh; do
    python run_bloom.py \
      --crosslingual \
      --hf_key=${key} \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr} \
      --num_shot=8
    python run_eval.py \
      --crosslingual \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done