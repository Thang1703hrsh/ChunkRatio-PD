#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Llama/ChunkRPO LoRA training launcher.
# ChunkRPO is value-head-free: no baseline head is used.

### OPTIONS

## MODEL
MODEL_CONFIG="llama_8b"

## LOSS
LOSS_NAME="ChunkRPO"
BREGMAN_NAME="sba"
BREGMAN_LAM="0.5"
BREGMAN_S="4.0"

## CHUNKRPO
ETA="1.0"
ALPHA="0.1"
N_MIN="4"
STANDARDIZE="false"
TAU_C="0.0"
RHO="0.0"
SALIENCY_TEMP="1.0"
MIN_CHUNK_TOKENS="4"
MAX_CHUNK_TOKENS="64"

## GENERAL CONFIG
BATCH_SIZE="32"
GRAD_ACCUM="2"
DATASETS_RAW="princeton-nlp/llama3-ultrafeedback-armorm"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
LR="5e-7"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="10.0"
OPTIMIZER="RMSprop"
SCHEDULER="cosine"
WARMUP_RATIO="0.05"
MIN_LOG_INTERVAL_SECS="1.0"

MAX_LENGTH="2048"
N_EPOCHS="1"
DO_FIRST_EVAL="true"
ACTIVATION_CHECKPOINTING="true"

python3 train.py \
  model=${MODEL_CONFIG} \
  loss=${LOSS_NAME} \
  loss.beta=0.1 \
  loss.bregman_loss.name=${BREGMAN_NAME} \
  loss.bregman_loss.lam=${BREGMAN_LAM} \
  loss.bregman_loss.s=${BREGMAN_S} \
  loss.chunkrpo.eta=${ETA} \
  loss.chunkrpo.alpha=${ALPHA} \
  loss.chunkrpo.n_min=${N_MIN} \
  loss.chunkrpo.standardize=${STANDARDIZE} \
  loss.chunkrpo.tau_c=${TAU_C} \
  loss.chunkrpo.rho=${RHO} \
  loss.chunkrpo.saliency_temp=${SALIENCY_TEMP} \
  loss.chunkrpo.min_chunk_tokens=${MIN_CHUNK_TOKENS} \
  loss.chunkrpo.max_chunk_tokens=${MAX_CHUNK_TOKENS} \
  datasets=${DATASETS_RAW} \
  dataset_train_split=${TRAIN_SPLIT} \
  dataset_test_split=${TEST_SPLIT} \
  batch_size=${BATCH_SIZE} \
  gradient_accumulation_steps=${GRAD_ACCUM} \
  lr=${LR} \
  weight_decay=${WEIGHT_DECAY} \
  max_grad_norm=${MAX_GRAD_NORM} \
  optimizer=${OPTIMIZER} \
  scheduler=${SCHEDULER} \
  warmup_ratio=${WARMUP_RATIO} \
  minimum_log_interval_secs=${MIN_LOG_INTERVAL_SECS} \
  max_length=${MAX_LENGTH} \
  n_epochs=${N_EPOCHS} \
  do_first_eval=${DO_FIRST_EVAL} \
  activation_checkpointing=${ACTIVATION_CHECKPOINTING} \
