#! /bin/bash

# conda activate skg
#export WANDB_API_KEY=2c824ff87dc5d0d79529da54b0eff68d1b587dac
export WANDB_PROJECT=mt5_movernight_few-shot
#export CUDA_LAUNCH_BLOCKING=1

source_lang=en
#export CUDA_VISIBLE_DEVICES=$1
export RUN_NAME=mt5-large_movernight_${source_lang}_lambda_new_monolingual_few-shot

python train.py --cfg XSP/mT5/monolingual_few-shot/mT5_movernight_${source_lang}.cfg \
--report_to wandb \
--run_name $RUN_NAME \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 300 \
--metric_for_best_model exact_match \
--greater_is_better true \
--save_strategy steps \
--save_steps 300 \
--save_total_limit 1 \
--load_best_model_at_end \
--gradient_accumulation_steps 16 \
--num_train_epochs 60 \
--adafactor true \
--learning_rate 5e-5 \
--do_train --do_eval --do_predict \
--predict_with_generate \
--output_dir output/$RUN_NAME \
--overwrite_output_dir \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--generation_num_beams 4 \
--generation_max_length 512 \
--input_max_length 1024

# nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_prefix.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 100 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1280 > $RUN_NAME.log 2>&1 &
