set -ex
#!/bin/bash

# volcengine.com
export NCCL_IB_PCI_RELAXED_ORDERING=1

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
MICRO_BATCH_SIZE=${9:-16}
GLOBAL_BATCH_SIZE=${10:-128}
SPLIT_RANK=${11:-0}
NUM_LAYER=${12:-0}
TRAIN_ITERS=${13:-220}
LOG_INTERVAL=${14:-100}
RUN_COMMIT=${15:-"e156d2f"}
DATA_PATH=${16:-"/dataset/source/dataset/loss_compara_content_sentence"}
VOCAB_FILE=${17:-"/dataset/source/dataset/bert-base-chinese-vocab.txt"}


SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_T5"
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

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah16_hs2304_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# nsys
#nsys profile --stats true --output ${LOG_FILENAME} \
CMD="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_t5.py \
       --tensor-model-parallel-size $MP \
           --pipeline-model-parallel-size $PP \
           --pipeline-model-parallel-split-rank $SPLIT_RANK \
       --num-layers $NUM_LAYER \
       --hidden-size 2304 \
       --num-attention-heads 16 \
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
           --vocab-extra-ids 100 "

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