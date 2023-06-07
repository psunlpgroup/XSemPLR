#! /bin/bash

arch=xlm-roberta-large
ckpt="/data/yfz5488/xsp/model/seq2seqPTR/output/matis_en_xlm-roberta-large_ckpt_20221227/lr0.00005_batch24"
vocab_path="/data/yfz5488/xsp/model/seq2seqPTR/data/matis/xlm-roberta-large/multilingual/sql/output_vocab.txt"
date=$(date +%Y%m%d)
dataset=matis
dataset_path=../../few-shot_dataset/
mr=sql
# We regard zero-shot as few-shot with no training. (we also run en as verification)
python data_preprocess.py --dataset=${dataset} --model=${arch} --dataset_path=${dataset_path}
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/

for language in en es de fr pt zh; do
    for learning_rate in 0; do
        for batch_size in 24; do
            EXPID=matis_${language}_${arch}_crosszero_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=$1 python ptrnet_bert.py \
                --dataset=MATIS \
                --max_seq_length=256 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev_test \
                --bert_model=${arch} \
                --bert_load_path=${ckpt} \
                --data_dir=data/${dataset}/${arch}/${language}/${mr} \
                --overwrite_output_vocab \
                --output_vocab=${vocab_path} \
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


# python data_preprocess.py --dataset=matis --model=xlm-roberta-large
# python data_preprocess.py --dataset=matis --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=matis --model=google/mt5-large