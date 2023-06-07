#! /bin/bash

arch=xlm-roberta-large
python data_preprocess.py --dataset=mgeoquery --model=${arch} --dataset_path=../../few-shot_dataset/

for language in 'en' 'de' 'th'; do
    for mr in lambda; do
        for learning_rate in 0.000001 0.000003 0.000005 0.00001 0.00002 0.00005 0.0001 0.0005;do
            for batch_size in 32; do
                EXPID=mgeoquery_${language}_${mr}_${arch}_few-shot_0803
                # echo $EXPID
                mkdir -p output/${EXPID}/lr${learning_rate}_batch${batch_size}
                CUDA_VISIBLE_DEVICES=1 python ptrnet_bert.py \
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
                # rm output/${EXPID}/lr${learning_rate}_batch${batch_size}/pytorch_model.bin
            done
        done
    done
done