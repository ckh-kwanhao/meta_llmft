#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# -----------------------------------------------------------------------------------------------------------------------
# run ICL experiments for MNLI
# -----------------------------------------------------------------------------------------------------------------------

export NCCL_DEBUG=INFO


bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh qqp 64 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh qqp 64 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh qqp 64 facebook/opt-125m 1 30000

bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh qqp 32 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh qqp 32 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh qqp 32 facebook/opt-125m 1 30000


bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh qqp 16 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh qqp 16 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh qqp 16 facebook/opt-125m 1 30000


bash $PROJECT_DIR/scripts/in_context/mnli/run_minimal.sh qqp 2 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_gpt3.sh qqp 2 facebook/opt-125m 1 30000
bash $PROJECT_DIR/scripts/in_context/mnli/run_eval_harness.sh qqp 2 facebook/opt-125m 1 30000