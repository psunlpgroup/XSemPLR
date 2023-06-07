#! /bin/bash

arch=xlm-roberta-large
date=$(date +%Y%m%d)
dataset=mgeoquery
dataset_path=../../translated_dataset/
type=translated

python data_preprocess.py --dataset=${dataset} --model=${arch} --dataset_path=${dataset_path}
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for language in 'de' 'th' 'zh' 'fa' 'el' 'id' 'sv'; do
    for mr in prolog sql; do
        for learning_rate in 0;do
            for batch_size in 32; do
                vocab_path=/data/yfz5488/xsp/model/seq2seqPTR/data/${dataset}/${arch}/multilingual/${mr}/output_vocab.txt
                ckpt=/data/yfz5488/xsp/model/seq2seqPTR/output/mgeoquery_en_xlm-roberta-large_${mr}_ckpt_20221228/lr0.00005_batch32
                EXPID=${dataset}_${language}_${arch}_${mr}_${type}_${date}
                # echo $EXPID
                mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
                CUDA_VISIBLE_DEVICES=$1 python ptrnet_bert.py \
                    --dataset=MGEOQUERY \
                    --max_seq_length=128 \
                    --smoothing=0.1 \
                    --use_decode_emb=0 \
                    --use_avg_span_extractor=1 \
                    --use_schema_token_mask=0 \
                    --mode=train \
                    --eval_on=dev_test \
                    --bert_model=${arch} \
                    --bert_load_path=${ckpt} \
                    --data_dir=data/mgeoquery/${arch}/${language}/${mr} \
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
done