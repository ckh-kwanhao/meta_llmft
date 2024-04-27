#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, bsz, num_gpus, model_name_or_path, port
#bash $PROJECT_DIR/scripts/vanilla_ft/cola/run.sh rte 64 32 1 facebook/opt-125m 60000
bash $PROJECT_DIR/scripts/vanilla_ft/mnli/run.sh mnli 64 32 0.5 1 1 0.0001 facebook/opt-350m 60000
