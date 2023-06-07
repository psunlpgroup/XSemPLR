#! /bin/bash
arch=xlm-roberta-large
date=$(date +%Y%m%d)
dataset=movernight
ckpt="/data/yfz5488/xsp/model/seq2seqPTR/output/movernight_en_xlm-roberta-large_ckpt_20221227/lr0.00005_batch32"
vocab_path="/data/yfz5488/xsp/model/seq2seqPTR/data/movernight/xlm-roberta-large/multilingual/lambda/output_vocab.txt"
dataset_path=../../few-shot_dataset/
mr=lambda

python data_preprocess.py --dataset=${dataset} --model=${arch} --dataset_path=${dataset_path}
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/

for language in de zh; do
    for learning_rate in 0.00005; do
        for batch_size in 32; do
            EXPID=${dataset}_${language}_${arch}_crossfew_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=7 python ptrnet_bert.py \
                --dataset=MOVERNIGHT \
                --max_seq_length=128 \
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