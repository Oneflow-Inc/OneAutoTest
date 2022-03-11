#!/bin/bash

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MP=${5:-1}
PP=${6:-1}
USE_FP16=${7:-false}
ACTIVATION_CHECKPOINT=${8:-false}
MICRO_BATCH_SIZE=${9:-2}
GLOBAL_BATCH_SIZE=${10:-16}
TRAIN_ITERS=${11:-320}
LOG_INTERVAL=${12:-100}
RUN_COMMIT=${13:-"e156d2f"}
DATA_PATH=${14:-"/home/ylkj/dataset/loss_compara_content_sentence"}
VOCAB_FILE=${15:-"/home/ylkj/dataset/bert-base-chinese-vocab.txt"}

SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_bert"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")

LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl24_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


CMD="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size $MP \
       --pipeline-model-parallel-size $PP \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters $TRAIN_ITERS \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval $LOG_INTERVAL \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 "

if $USE_FP16; then
    CMD+=" \
      --fp16 "
fi

if $ACTIVATION_CHECKPOINT; then
    CMD+=" \
      --activations-checkpoint-method uniform "
    if [ ${MP} -gt 1 ];then
        CMD+=" \
          --distribute-checkpointed-activations "
    fi
fi


echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log


