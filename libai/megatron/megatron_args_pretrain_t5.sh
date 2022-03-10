#!/bin/bash

NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DATA_PATH=${5:-"/dataset/loss_compara_content_sentence"}
VOCAB_FILE=${6:-"/dataset/bert-base-chinese-vocab.txt"}
MP=${7:-1}
PP=${8:-1}
MICRO_BATCH_SIZE=${9:-16}
GLOBAL_BATCH_SIZE=${10:-128}
TRAIN_ITERS=${11:-320}
LOG_INTERVAL=${12:-100}
RUN_COMMIT=${13:-"e156d2f"}

SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_T5"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/${NUM_NODES}n${DEVICE_NUM_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl12_nah12_hs768_fp16_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NUM_NODES}n${DEVICE_NUM_PER_NODE}g_${RUN_COMMIT}_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size 2 \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --kv-channels 64 \
       --ffn-hidden-size 3072 \
       --encoder-seq-length 512 \
       --decoder-seq-length 128 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --max-position-embeddings 512 \
       --train-iters $TRAIN_ITERS \
       --lr-decay-iters 1000000 \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-warmup-fraction .01 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval $LOG_INTERVAL \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16  \
       --vocab-extra-ids 100 2>&1 | tee ${LOG_FILENAME}.log
