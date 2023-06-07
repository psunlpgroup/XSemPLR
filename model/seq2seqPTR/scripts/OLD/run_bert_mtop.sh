#! /bin/bash

###### MTOP
# data preprocessing
#python data_preprocess.py --dataset=mtop --model=bert-base-cased --output_dir=data

# train and eval
mkdir -p output/mtop_bert
CUDA_VISIBLE_DEVICES=0 python ptrnet_bert.py \
    --dataset=MTOP \
    --mode=train \
    --eval_on=dev_test \
    --bert_model=bert-base-cased \
    --data_dir=data/mtop/bert-base-cased/en \
    --output_vocab_f=data/mtop/bert-base-cased/en/output_vocab.txt \
    --output_dir=output/mtop_bert \
    --num_train_epochs=30


######## Old
# train and eval
# CUDA_VISIBLE_DEVICES=1 python ptrnet_bert.py \
#     --dataset=MTOP \
#     --mode=train \
#     --eval_on=dev_test \
#     --train_dir=data/mtop/train \
#     --dev_dir=data/mtop/dev \
#     --test_dir=data/mtop/test.py \
#     --output_vocab_f=data/mtop/output_vocab.txt \
#     --output_dir=output/mtop_test \
#     --num_train_epochs=30

###### TOP
# train and eval
# CUDA_VISIBLE_DEVICES=2 python ptrnet_bert.py \
#     --dataset=TOP \
#     --mode=train \
#     --eval_on=test.py \
#     --train_dir=data/woUnsupported/train \
#     --dev_dir=data/woUnsupported/valid \
#     --test_dir=data/woUnsupported/test.py \
#     --output_vocab_f=data/woUnsupported/output_vocab.txt \
#     --output_dir=output/top_test_0407 \
#     --num_train_epochs=200