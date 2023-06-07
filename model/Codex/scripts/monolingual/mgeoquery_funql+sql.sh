#! /bin/bash

key=YOUR_API_KEY
dataset=mgeoquery
data_folder=../../dataset/

for language in 'en' 'de' 'th' 'zh' 'fa' 'el' 'id' 'sv'; do
    for mr in funql sql; do
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
done