#! /bin/bash

dataset=mspider
data_folder=../../dataset/
mr=sql
key=YOUR_API_KEY
for language in vi zh; do
    python run_bloom.py \
      --crosslingual \
      --hf_key=${key} \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr} \
      --num_shot=1

    python run_eval.py \
      --crosslingual \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done


