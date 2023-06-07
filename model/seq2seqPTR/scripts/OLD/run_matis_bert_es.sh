#! /bin/bash
export PYTORCH_TRANSFORMERS_CACHE='/data/yfz5488/.cache/huggingface'
arch=bert-base-multilingual-cased
date=0624
# python data_preprocess.py --dataset=matis --model=${arch}
for language in es; do
    for learning_rate in 0.00005; do
        for batch_size in 64; do
            EXPID=matis_${language}_${arch}_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=2 python ptrnet_bert.py \
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
                --dev_batch_size 128 \
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