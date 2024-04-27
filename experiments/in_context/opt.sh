#!/usr/bin/env bash

#original
#export PROJECT_DIR=/home/mmosbach/projects/llmft
#source $PROJECT_DIR/scripts/misc/setup.sh

# -----------------------------------------------------------------------------------------------------------------------
# run ICL experiments for MNLI
# -----------------------------------------------------------------------------------------------------------------------

#copied
export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, bsz, num_gpus, model_name_or_path, port
#bash $PROJECT_DIR/scripts/vanilla_ft/cola/run.sh rte 64 32 1 facebook/opt-125m 60000
#bash $PROJECT_DIR/scripts/vanilla_ft/mnli/run.sh rte 64 32 0.5 1 1 0.0001 facebook/opt-350m 50000




export NCCL_DEBUG=INFO

bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 2 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 2 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 2 facebook/opt-30b 1 60000

bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 16 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 16 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 16 facebook/opt-30b 1 60000

bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 32 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 32 facebook/opt-30b 1 60000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 32 facebook/opt-30b 1 60000
