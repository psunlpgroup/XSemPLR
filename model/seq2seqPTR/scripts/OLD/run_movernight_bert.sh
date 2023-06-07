#! /bin/bash

export PYTORCH_TRANSFORMERS_CACHE='/data/yfz5488/.cache/huggingface'
arch=bert-base-multilingual-cased
# arch=bert-base-cased
date=0623

python data_preprocess.py --dataset=movernight --model=${arch}

for language in 'en' 'de' 'zh'; do
    for learning_rate in 0.00005 0.0001; do
        for batch_size in 32; do
            EXPID=movernight_${language}_${arch}_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=2 /data/yfz5488/anaconda3/envs/xspptr/bin/python ptrnet_bert.py \
                --dataset=MOVERNIGHT \
                --max_seq_length=128 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev_test \
                --bert_model=${arch} \
                --data_dir=data/movernight/${arch}/${language}/lambda \
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

# python data_preprocess.py --dataset=movernight --model=xlm-roberta-large
# python data_preprocess.py --dataset=movernight --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=movernight --model=google/mt5-large