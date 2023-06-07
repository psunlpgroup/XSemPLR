#! /bin/bash

export WANDB_PROJECT=mt5_matis
export CUDA_VISIBLE_DEVICES=0
lang=zh
export RUN_NAME=mt5-large_matis_${lang}_sql

python train.py --cfg XSP/mT5_matis_${lang}.cfg \
--report_to wandb \
--run_name $RUN_NAME \
--logging_strategy steps \
--logging_first_step true \
--logging_steps 4 \
--evaluation_strategy steps \
--eval_steps 800 \
--metric_for_best_model exact_match \
--greater_is_better true \
--save_strategy steps \
--save_steps 800 \
--save_total_limit 15 \
--load_best_model_at_end \
--gradient_accumulation_steps 16 \
--num_train_epochs 80 \
--adafactor true \
--learning_rate 5e-5 \
--do_train --do_eval \
--do_predict --predict_with_generate \
--output_dir output/$RUN_NAME \
--overwrite_output_dir \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--generation_num_beams 4 \
--generation_max_length 512 \
--input_max_length 1280

# nohup python -m torch.distributed.launch --nproc_per_node 8 --master_port 1234 train.py --cfg META_TUNING/T5_prefix.cfg --report_to wandb --run_name $RUN_NAME --logging_strategy steps --logging_first_step true --logging_steps 4 --evaluation_strategy steps --eval_steps 500 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 500 --save_total_limit 15 --load_best_model_at_end --gradient_accumulation_steps 16 --num_train_epochs 100 --adafactor true --learning_rate 5e-5 --do_train --do_eval --do_predict --predict_with_generate --output_dir output/$RUN_NAME --overwrite_output_dir --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --generation_num_beams 4 --generation_max_length 512 --input_max_length 1280 > $RUN_NAME.log 2>&1 &

#    “run_name”: “mt5-large-cspider”,
#    “model_name_or_path”: “google/mt5-large”,
#    “dataset”: “cspider”,
#    “dataset_name”: “cspider”,
#    “mlm_dataset”: “”,
#    “mlm_dataset_name”: “”,
#    “source_prefix”: “”,
#    “schema_serialization_type”: “peteshaw”,
#    “schema_serialization_randomized”: false,
#    “schema_serialization_with_db_id”: true,
#    “schema_serialization_with_db_content”: true,
#    “normalize_query”: true,
#    “target_with_db_id”: true,
#    “output_dir”: “spider_mt5-large_train_cspider”,
#    “cache_dir”: “transformers_cache”,
#    “do_train”: true,
#    “do_eval”: true,
#    “fp16": false,
#    “num_train_epochs”: 3072,
#    “per_device_train_batch_size”: 2,
#    “per_device_eval_batch_size”: 2,
#    “gradient_accumulation_steps”: 256,
#    “label_smoothing_factor”: 0.0,
#    “learning_rate”: 1e-4,
#    “adafactor”: true,
#    “adam_eps”: 1e-6,
#    “lr_scheduler_type”: “constant”,
#    “warmup_ratio”: 0.0,
#    “warmup_steps”: 0,
#    “seed”: 1,
#    “report_to”: [“wandb”],
#    “logging_strategy”: “steps”,
#    “logging_first_step”: true,
#    “logging_steps”: 4,
#    “load_best_model_at_end”: true,
#    “metric_for_best_model”: “exact_match”,
#    “greater_is_better”: true,
#    “save_total_limit”: 128,
#    “save_steps”: 64,
#    “evaluation_strategy”: “steps”,
#    “eval_steps”: 64,
#    “predict_with_generate”: true,
#    “num_beams”: 1,
#    “num_beam_groups”: 1,
#    “use_picard”: false