#! /bin/bash

# conda activate skg
#export WANDB_API_KEY=99356b982e99520d418737dbb0198e07dbdb8abe
export WANDB_PROJECT=mt5-large_mgeoquery
export CUDA_LAUNCH_BLOCKING=1

source_lang=sv
target_lang=funql
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=mt5-large_mgeoquery_${source_lang}_${target_lang}

python train.py --cfg XSP/mT5_mgeoquery_${source_lang}_${target_lang}.cfg \
--report_to wandb \
--run_name $RUN_NAME \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 100 \
--metric_for_best_model exact_match \
--greater_is_better true \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 18 \
--load_best_model_at_end \
--gradient_accumulation_steps 16 \
--num_train_epochs 100  \
--adafactor true \
--learning_rate 5e-5 \
--do_train --do_eval --do_predict \
--predict_with_generate \
--output_dir output/$RUN_NAME \
--overwrite_output_dir \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--generation_num_beams 6 \
--generation_max_length 512 \
--input_max_length 1280

# nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_prefix.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 100 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1280 > $RUN_NAME.log 2>&1 &
