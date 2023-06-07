#! /bin/bash

export WANDB_PROJECT=mt5-large_mconala
export CUDA_VISIBLE_DEVICES=$1

lang=es
export RUN_NAME=mt5-large_mconala_${lang}

python train.py --cfg XSP/mT5_mconala_${lang}.cfg \
  --report_to wandb \
  --run_name $RUN_NAME \
  --logging_strategy steps \
  --logging_first_step true \
  --logging_steps 4 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --metric_for_best_model exact_match \
  --greater_is_better true \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --gradient_accumulation_steps 16 \
  --resume_from_checkpoint /home/yfz5488/xsp/model/UniPSP/output/mt5-large_mconala_en \
  --num_train_epochs 100 \
  --adafactor true \
  --learning_rate 5e-5 \
  --do_predict \
  --predict_with_generate \
  --output_dir output/$RUN_NAME \
  --overwrite_output_dir \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --generation_num_beams 6 \
  --generation_max_length 512 \
  --input_max_length 1280

# nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_prefix.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 100 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1280 > $RUN_NAME.log 2>&1 &
