#! /bin/bash

export PYTORCH_TRANSFORMERS_CACHE='/data/yfz5488/.cache/huggingface'
arch=bert-base-multilingual-cased
date=0624
python data_preprocess.py --dataset=mtop --model=${arch}
for language in de fr th es hi en; do
    # for learning_rate in 0.00002; do
    for learning_rate in 0.0001;do
        for batch_size in 64; do
EXPID=mtop_${language}_${arch}_${date}
# echo $EXPID
mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
CUDA_VISIBLE_DEVICES=1 python ptrnet_bert.py \
    --dataset=MTOP \
    --max_seq_length=128 \
    --smoothing=0.1 \
    --use_decode_emb=0 \
    --use_avg_span_extractor=1 \
    --use_schema_token_mask=0 \
    --mode=train \
    --eval_on=dev_test \
    --bert_model=${arch} \
    --data_dir=data/mtop/${arch}/${language} \
    --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
    --num_train_epochs=100 \
    --learning_rate=${learning_rate} \
    --train_batch_size=${batch_size} \
    --bert_lr=${learning_rate} \
    --wandb_project=${EXPID}
# rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
        done
    done
done

# python data_preprocess.py --dataset=mtop --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mtop --model=google/mt5-large