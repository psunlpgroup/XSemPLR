#! /bin/bash

key=YOUR_API_KEY
dataset=mschema2qa
data_folder=../../dataset/
mr=thingtalk

for language in 'en' 'de' 'es' 'ar' 'fi' 'it' 'tr' 'zh' 'pl' 'fa' 'ja'; do
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