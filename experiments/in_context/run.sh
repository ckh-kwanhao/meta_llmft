#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# -----------------------------------------------------------------------------------------------------------------------
# run ICL experiments for MNLI
# -----------------------------------------------------------------------------------------------------------------------

export NCCL_DEBUG=INFO


# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 64 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 64 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 64 facebook/opt-125m 1 30000

# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 32 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 32 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 32 facebook/opt-125m 1 30000


# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 16 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 16 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 16 facebook/opt-125m 1 30000


# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 2 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 2 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 2 facebook/opt-125m 1 30000


# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 16 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 2 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 32 facebook/opt-125m 1 30000

# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 16 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 16 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh mnli 2 facebook/opt-125m 1 30000
# bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh mnli 2 facebook/opt-125m 1 30000

bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh mnli 4 facebook/opt-125m 1 30000
