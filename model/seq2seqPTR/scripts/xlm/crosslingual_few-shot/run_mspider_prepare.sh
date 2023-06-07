#! /bin/bash

arch=xlm-roberta-large
date=$(date +%Y%m%d)
dataset=mspider
dataset_path=../../dataset/

python preprocess.py --dataset=spider --remove_from
python data_preprocess.py --dataset=${dataset} --model=${arch} --dataset_path=${dataset_path}
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for learning_rate in 0.00005;do
    for batch_size in 8;do
        for language in en; do
            vocab_path=/data/yfz5488/xsp/model/seq2seqPTR/data/${dataset}/${arch}/multilingual/output_vocab.txt
            EXPID=${dataset}_${language}_${arch}_ckpt_${date}
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
                --data_dir=data/${dataset}/${arch}/${language} \
                --overwrite_output_vocab \
                --output_vocab=${vocab_path} \
                --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                --num_train_epochs=60 \
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