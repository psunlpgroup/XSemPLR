#! /bin/bash

#! /bin/bash

key=YOUR_API_KEY
dataset=mspider
data_folder=../../dataset/
mr=sql

for language in en vi zh; do
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


