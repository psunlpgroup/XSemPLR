#! /bin/bash

arch=xlm-roberta-large
date=20230101
python preprocess.py --dataset=spider --remove_from
python data_preprocess.py --dataset=mspider --model=${arch} --dataset_path=../../few-shot_dataset/
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001;do
    for batch_size in 8;do
        for language in en zh vi; do
            EXPID=mspider_${language}_${arch}_monofew_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
#            CUDA_VISIBLE_DEVICES=$1 python ptrnet_bert.py \
#                --dataset=MSPIDER \
#                --max_seq_length=256 \
#                --smoothing=0.1 \
#                --use_decode_emb=0 \
#                --use_avg_span_extractor=1 \
#                --use_schema_token_mask=0 \
#                --mode=train \
#                --eval_on=dev \
#                --bert_model=${arch} \
#                --data_dir=data/mspider/${arch}/${language} \
#                --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
#                --num_train_epochs=60 \
#                --learning_rate=${learning_rate} \
#                --train_batch_size=${batch_size} \
#                --bert_lr=${learning_rate} \
#                --wandb_project=${EXPID}
            python postprocess_eval.py --dataset=spider --split=dev --pred_file output/${EXPID}/lr${learning_rate}_batch${batch_size}/dev_output.json --remove_from
            # rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
        done
    done
done


# python data_preprocess.py --dataset=mspider --model=xlm-roberta-large
# python data_preprocess.py --dataset=mspider --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mspider --model=google/mt5-large