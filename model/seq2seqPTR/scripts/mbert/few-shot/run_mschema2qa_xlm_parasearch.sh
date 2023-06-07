#! /bin/bash

date=0805
arch=xlm-roberta-large
python data_preprocess.py --dataset=mschema2qa --model=${arch} --dataset_path=../../few-shot_dataset/

for language in 'en' 'de' 'es' 'ar' 'fi' 'it' 'tr' 'zh' 'pl' 'fa' 'ja'; do
    for learning_rate in 0.000005 0.00001 0.00002 0.00005 0.0001; do
        for batch_size in 32; do
            EXPID=mschema2qa_${language}_${arch}_${date}
            mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
            CUDA_VISIBLE_DEVICES=2 python ptrnet_bert.py \
                --dataset=MSCHEMA2QA \
                --max_seq_length=128 \
                --smoothing=0.1 \
                --use_decode_emb=0 \
                --use_avg_span_extractor=1 \
                --use_schema_token_mask=0 \
                --mode=train \
                --eval_on=test \
                --bert_model=${arch} \
                --data_dir=data/mschema2qa/${arch}/${language}/thingtalk \
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

# python data_preprocess.py --dataset=mschema2qa --model=xlm-roberta-large
# python data_preprocess.py --dataset=mschema2qa --model=facebook/mbart-large-50
# python data_preprocess.py --dataset=mschema2qa --model=google/mt5-large