#! /bin/bash

arch=xlm-roberta-large
date=$(date +%Y%m%d)
dataset_path=../../translated_dataset/
type=translated
python preprocess.py --dataset=spider --remove_from
python data_preprocess.py --dataset=mspider --model=${arch} --dataset_path=../../few-shot_dataset/
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for learning_rate in 0;do
    for batch_size in 24;do
        for language in en zh vi; do
            EXPID=mspider_${language}_${arch}_crosszero_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=$1 python ptrnet_bert.py \
                --dataset=MSPIDER \
                --max_seq_length=256 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev \
                --bert_model=${arch} \
                --data_dir=data/mspider/${arch}/${language} \
                --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                --num_train_epochs=1 \
                --learning_rate=${learning_rate} \
                --train_batch_size=${batch_size} \
                --bert_lr=${learning_rate} \
                --wandb_project=${EXPID}
            # rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
        done
    done
done

python postprocess_eval.py --dataset=spider --split=dev --pred_file output/${EXPID}/lr${learning_rate}_batch${batch_size}/valid_use_predicted_queries_predictions.json --remove_from

# python data_preprocess.py --dataset=mspider --model=xlm-roberta-large
# python data_preprocess.py --dataset=mspider --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mspider --model=google/mt5-large