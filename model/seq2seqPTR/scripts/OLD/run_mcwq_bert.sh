#! /bin/bash

export PYTORCH_TRANSFORMERS_CACHE='/data/yfz5488/.cache/huggingface'
arch=bert-base-multilingual-cased
# arch=bert-base-cased
date=0623

python data_preprocess.py --dataset=mcwq --model=${arch}

for language in en he kn zh; do
    for learning_rate in 0.00005 0.0001;do
        for batch_size in 32;do
            EXPID=mcwq_${language}_${arch}_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=0 /data/yfz5488/anaconda3/envs/xspptr/bin/python ptrnet_bert.py \
                --dataset=MCWQ \
                --max_seq_length=128 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev_test \
                --bert_model=${arch} \
                --data_dir=data/mcwq/${arch}/${language}/sparql \
                --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                --num_train_epochs=100 \
                --learning_rate=${learning_rate} \
                --train_batch_size=${batch_size} \
                --bert_lr=${learning_rate} \
                --wandb_project=${EXPID}
            rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
        done
    done
done


# python data_preprocess.py --dataset=mcwq --model=xlm-roberta-large
# python data_preprocess.py --dataset=mcwq --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mcwq --model=google/mt5-large