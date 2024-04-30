#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/run_kd.sh mnli 512 1 0.5 4 1 2e-5 facebook/opt-125m 40000
#bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/run_kd.sh mnli 128 10 0.5 4 1 1e-5 facebook/opt-125m 40000

