#! /bin/bash

arch=xlm-roberta-large
date=0805
python data_preprocess.py --dataset=matis --model=${arch} --dataset_path=../../few-shot_dataset/

for language in en de fr th es hi; do
    for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001; do
        for batch_size in 32; do
            EXPID=matis_${language}_${arch}_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=0 python ptrnet_bert.py \
                --dataset=MATIS \
                --max_seq_length=512 \
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