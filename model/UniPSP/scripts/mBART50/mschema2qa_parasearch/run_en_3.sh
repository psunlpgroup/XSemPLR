#! /bin/bash
source_lang=en
target_lang=thingtalk
para_id=3
export WANDB_PROJECT=mBART50-large_mschema2qa
export CUDA_VISIBLE_DEVICES=$1
export RUN_NAME=mBART50-large_mschema2qa_${source_lang}_${para_id}

python train.py --cfg XSP/mBART50/mschema2qa/en_thingtalk.cfg \
--report_to wandb \
--run_name $RUN_NAME \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 100 \
--metric_for_best_model exact_match \
--greater_is_better true \
--save_strategy steps --save_steps 100 \
--save_total_limit 3 \
--load_best_model_at_end \
--gradient_accumulation_steps 8 \
--num_train_epochs 50  \
--adam_eps 1e-6 \
--learning_rate 3e-5 \
--lr_scheduler_type constant \
--do_train --do_eval --do_predict \
--predict_with_generate \
--output_dir output/$RUN_NAME \
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 8 \
--generation_num_beams 6 \
--generation_max_length 512 \
--input_max_length 1024
