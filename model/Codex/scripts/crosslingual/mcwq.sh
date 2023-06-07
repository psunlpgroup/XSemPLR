#! /bin/bash

key=YOUR_API_KEY
dataset=mcwq
data_folder=../../dataset/
mr=sparql

for language in en he kn zh; do
    python run_codex.py \
      --openai_key=${key} \
      --crosslingual \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr} \
      --num_shot=8
    python run_eval.py \
      --dataset=${dataset} \
      --crosslingual \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done