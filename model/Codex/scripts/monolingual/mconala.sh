#! /bin/bash

key=YOUR_API_KEY
dataset=mconala
data_folder=../../dataset/
mr=python

for language in en; do
    python run_codex.py \
      --openai_key=${key} \
      --dataset=${dataset} \
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