#! /bin/bash

###### mgeoquery
# python data_preprocess.py --dataset=mgeoquery --model=facebook/bart-large

for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001;do
    for batch_size in 16 32 64 128;do
        mkdir -p output/mgeoquery_funql_bart_0802/lr${learning_rate}_batch${batch_size}
        CUDA_VISIBLE_DEVICES=3 python ptrnet_bert.py \
            --dataset=MGEOQUERY \
            --max_seq_length=128 \
            --smoothing=0.1 \
            --use_decode_emb=0 \
            --use_avg_span_extractor=1 \
            --use_schema_token_mask=0 \
            --mode=train \
            --eval_on=dev_test \
            --bert_model=facebook/bart-large \
            --data_dir=data/mgeoquery/facebook_bart-large/en/funql \
            --output_dir=output/mgeoquery_funql_bart_0802/lr${learning_rate}_batch${batch_size} \
            --num_train_epochs=300 \
            --learning_rate=${learning_rate} \
            --train_batch_size=${batch_size} \
            --bert_lr=${learning_rate} \
            --wandb_project=xsp-mgeoquery-funql-bart
        # exit
    done
done