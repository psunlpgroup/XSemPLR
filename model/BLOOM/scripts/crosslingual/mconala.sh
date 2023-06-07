#! /bin/bash

dataset=mconala
data_folder=../../dataset/
mr=python
key=YOUR_API_KEY

#for language in mix; do # we use mixed language to denote mconala special case
#    python run_bloom.py \
#      --crosslingual \
#      --dataset=${dataset} \
#      --hf_key=${key} \
#      --data_folder=${data_folder} \
#      --language=${language} \
#      --mr=${mr} \
#      --num_shot=8 \
#      --max_attempt=2
#done

for language in ru es ja; do
  python run_eval.py \
      --crosslingual \
      --dataset=${dataset} \
      --data_folder=${data_folder} \
      --language=${language} \
      --mr=${mr}
done