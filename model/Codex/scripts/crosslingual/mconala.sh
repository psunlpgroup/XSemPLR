#! /bin/bash

key=YOUR_API_KEY
dataset=mconala
data_folder=../../dataset/
mr=python

for language in es ja ru; do
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