#! /bin/bash

dataset=mtop
data_folder=../../dataset/
mr=slot_intent
key=YOUR_API_KEY

for language in de fr th es hi; do
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