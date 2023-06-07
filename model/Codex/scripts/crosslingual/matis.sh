#! /bin/bash

key=YOUR_API_KEY
dataset=matis
data_folder=../../dataset/
mr=sql

for language in en es de fr pt zh; do
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


