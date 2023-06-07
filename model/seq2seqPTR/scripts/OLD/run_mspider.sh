#! /bin/bash

# python preprocess.py --dataset=spider --remove_from
# python data_preprocess.py --dataset=mspider --model=bert-base-multilingual-cased
# python postprocess_eval.py --dataset=spider --split=dev --pred_file $LOGDIR/valid_use_predicted_queries_predictions.json --remove_from
for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001;do
    for batch_size in 16 32 64 128;do
        for language in 'en', 'zh', 'vi'; do
            EXPID=mspider_${language}_mbert_0824
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=0 python ptrnet_bert.py \
                --dataset=MSPIDER \
                --max_seq_length=512 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=dev \
                --bert_model=bert-base-multilingual-cased \
                --data_dir=data/mspider/bert-base-multilingual-cased/${language} \
                --output_dir=output/${EXPID}/lr${learning_rate}_batch${batch_size} \
                --num_train_epochs=30 \
                --learning_rate=${learning_rate} \
                --train_batch_size=${batch_size} \
                --bert_lr=${learning_rate} \
                --wandb_project=${EXPID}
            rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
        done
    done
done

# python data_preprocess.py --dataset=mspider --model=xlm-roberta-large
# python data_preprocess.py --dataset=mspider --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mspider --model=google/mt5-large