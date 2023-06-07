#! /bin/bash

date=1205
arch=xlm-roberta-large
python data_preprocess.py --dataset=matis --model=${arch} --dataset_path=../../multilingual_dataset/
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/

for language in 'multilingual'; do
    for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001; do
        for batch_size in 24; do
            EXPID=matis_${language}_${arch}_multilingual_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=6 python ptrnet_bert.py \
                --dataset=MATIS \
                --max_seq_length=256 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev_test \
                --bert_model=${arch} \
                --data_dir=data/matis/${arch}/${language}/sql \
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


# python data_preprocess.py --dataset=matis --model=xlm-roberta-large
# python data_preprocess.py --dataset=matis --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=matis --model=google/mt5-large