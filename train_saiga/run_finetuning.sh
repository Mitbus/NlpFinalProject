#!/usr/bin/env bash

CUDA_DEVICES="0"
NPROC_PER_NODE=$(echo $CUDA_DEVICES | tr "," "\n" | wc -l)

BASE_DIR="/home/jovyan/tsukanov/test_calc/lora_train/lora-saiga"

SCRIPT_ARGS=" \
--base-model 'Open-Orca/Mistral-7B-OpenOrca' \
--train_data_path '$BASE_DIR/train_data.json' \
--test_data_path '$BASE_DIR/test_data.json' \
--output_dir '$BASE_DIR/models/v3' \
--batch_size 64 \
--micro_batch_size 32 \
--num_epochs 10 \
--learning_rate 5e-5 \
--cutoff_len 688 \
--warmup_steps 50 \
--logging_steps 10 \
--eval_steps 91 \
--save_steps 91 \
--lora_r 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
--train_on_inputs \
--resume_from_checkpoint '$BASE_DIR/models/saiga_mistal_7b_lora' \
--prompt_template_name 'saiga_short' \
"
# --cutoff_len 848 \
# --prompt_template_name 'saiga_long' \

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES WORLD_SIZE=$NPROC_PER_NODE /home/user/conda/envs/py39_env/bin/python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE finetune.py $SCRIPT_ARGS

nohup /home/user/conda/envs/py39_env/bin/python finetune.py $SCRIPT_ARGS &> $BASE_DIR/logs/v3.txt & disown
