#!/bin/bash

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DATA_PATH=${5:-"/dataset/loss_compara_content_sentence"}
VOCAB_FILE=${6:-"/dataset/gpt2-vocab.json"}
MERGE_FILE=${7:-"/dataset/gpt2-merges.txt"}
MP=${8:-1}
PP=${9:-1}
MICRO_BATCH_SIZE=${10:-16}
GLOBAL_BATCH_SIZE=${11:-128}
TRAIN_ITERS=${12:-320}
LOG_INTERVAL=${13:-100}
RUN_COMMIT=${14:-"e156d2f"}

SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_GPT2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/${NUM_NODES}n${DEVICE_NUM_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl24_nah16_hs1024_fp16_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NUM_NODES}n${DEVICE_NUM_PER_NODE}g_${RUN_COMMIT}_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $MP \
       --pipeline-model-parallel-size $PP \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters $TRAIN_ITERS \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --log-interval $LOG_INTERVAL \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 2>&1 | tee ${LOG_FILENAME}.log
