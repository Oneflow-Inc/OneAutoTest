#!/bin/bash

# volcengine.com
#export NCCL_IB_PCI_RELAXED_ORDERING=1

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
NUM_LAYER=${11:-24}
TRAIN_ITERS=${12:-220}
LOG_INTERVAL=${13:-100}
RUN_COMMIT=${14:-"e156d2f"}
DATA_PATH=${15:-"/path/to/loss_compara_content_sentence"}
VOCAB_FILE=${16:-"/path/to/gpt2-vocab.json"}
MERGE_FILE=${17:-"/path/to/gpt2-merges.txt"}


SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

# log
#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG=cudnn.log
#export GLOG_v=3

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# nsys
#nsys profile --stats true --output ${LOG_FILENAME} \
CMD="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size $MP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYER \
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

