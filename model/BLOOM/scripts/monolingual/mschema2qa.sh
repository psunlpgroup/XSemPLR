#! /bin/bash

dataset=mschema2qa
data_folder=../../dataset/
mr=thingtalk
key=YOUR_API_KEY

for language in 'en' 'de' 'es' 'ar' 'fi' 'it' 'tr' 'zh' 'pl' 'fa' 'ja'; do
    python run_bloom.py \
      --hf_key=${key} \
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