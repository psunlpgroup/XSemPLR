#! /bin/bash

arch=xlm-roberta-large
date=0101
python data_preprocess.py --dataset=mgeoquery --model=${arch} --dataset_path=../../dataset/
export TRANSFORMERS_CACHE=/data/yfz5488/.hf_cache/
for language in 'en' 'de' 'el' 'fa' 'id' 'sv' 'th' 'zh'; do
    for mr in funql prolog lambda sql; do
        for learning_rate in 0.00002 0.00005;do
            for batch_size in 32; do
                EXPID=mgeoquery_${language}_${mr}_${arch}_${date}
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
                    --data_dir=data/mgeoquery/${arch}/${language}/${mr} \
                    --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                    --num_train_epochs=300 \
                    --learning_rate=${learning_rate} \
                    --train_batch_size=${batch_size} \
                    --bert_lr=${learning_rate} \
                    --wandb_project=${EXPID}
                rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
            done
        done
    done
done

# python data_preprocess.py --dataset=mgeoquery --model=xlm-roberta-large
# python data_preprocess.py --dataset=mgeoquery --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mgeoquery --model=google/mt5-large