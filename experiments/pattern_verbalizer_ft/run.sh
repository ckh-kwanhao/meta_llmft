#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
#bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/run.sh mnli 128 40 0.5 1 8 1e-5 facebook/opt-13b 60000
#bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/cola/run.sh mnli 128 40 0.5 1 1 1e-5 facebook/opt-125m 50000
# bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/run.sh mnli 128 30 0.5 1 1 1e-5 facebook/opt-350m 60000
#bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/qqp/run.sh qqp 128 40 0.5 1 1 1e-5 facebook/opt-1.3b 60000
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/run.sh mnli 512 1 0.5 1 1 1e-4 facebook/opt-125m 60000
