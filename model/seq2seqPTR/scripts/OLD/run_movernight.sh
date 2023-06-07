#! /bin/bash

# python data_preprocess.py --dataset=movernight --model=bert-base-multilingual-cased
for language in 'en' 'de' 'zh'; do
    for learning_rate in 0.00001 0.00002 0.00005 0.0001 0.0005; do
        for batch_size in 32; do
            EXPID=movernight_${language}_mbert_0826
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=0 python ptrnet_bert.py \
                --dataset=MOVERNIGHT \
                --max_seq_length=128 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev_test \
                --bert_model=bert-base-multilingual-cased \
                --data_dir=data/movernight/bert-base-multilingual-cased/${language} \
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

# python data_preprocess.py --dataset=movernight --model=xlm-roberta-large
# python data_preprocess.py --dataset=movernight --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=movernight --model=google/mt5-large