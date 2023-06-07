#! /bin/bash

arch=xlm-roberta-large
date=$(date +%Y%m%d)
python data_preprocess.py --dataset=mgeoquery --model=${arch} --dataset_path=../../few-shot_dataset/
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for language in 'en' 'de' 'th' 'zh' 'fa' 'el' 'id' 'sv'; do
    for mr in prolog sql; do
        for learning_rate in 0.00001;do
            for batch_size in 32; do
                EXPID=mgeoquery_${language}_${mr}_${arch}_monofew_${date}
                # echo $EXPID
                mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
                CUDA_VISIBLE_DEVICES=7 python ptrnet_bert.py \
                    --dataset=MGEOQUERY \
                    --max_seq_length=128 \
                    --smoothing=0.1 \
                    --use_decode_emb=0 \
                    --use_avg_span_extractor=1 \
                    --use_schema_token_mask=0 \
                    --mode=train \
                    --eval_on=dev_test \
                    --bert_model=${arch} \
                    --data_dir=data/mgeoquery/${arch}/${language}/${mr} \
                    --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                    --num_train_epochs=300 \
                    --learning_rate=${learning_rate} \
                    --train_batch_size=${batch_size} \
                    --bert_lr=${learning_rate} \
                    --wandb_project=${EXPID}
                # rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
            done
        done
    done
done