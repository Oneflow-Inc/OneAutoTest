#!/bin/bash

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_GDR_LEVEL=2
# 安装 TCCL 之后不需要 NCCL TOPO 文件 comment 这行
# export NCCL_TOPO_FILE=/data_32T/home/workspace/nccl-tests/nccl_topo_a800_1.6t.xml
export NCCL_IB_QPS_PER_CONNECTION=4

# volcengine.com
#export NCCL_IB_PCI_RELAXED_ORDERING=1
NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
# Change for multinode config
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=6000
MP=${5:-1}
PP=${6:-1}
GRAPH_ENABLED=${7:-true}  
USE_FP16=${8:-false}  
ACTIVATION_CHECKPOINT=${9:-true}
MICRO_BATCH_SIZE=${10:-2}
ACC=${11:-1}
ZERO_ENABLE=${12:-false} 
ZERO_STAGE=${13:-0} 
TRAIN_ITERS=${14:-220}
LOG_PERIOD=${15:-100}
NUM_LAYER=${16:-24}
NUM_ATT_HEADS=${17:-16}
HIDDEN_SIZE=${18:-768}
INTERMEDIATE_SIZE=${19:-3072}
HEAD_SIZE=${20:-64} 
SAVE_MODEL=${21:-false} 
UNSET_DROPOUT=${22:-false} 
DATA_PATH=${23:-"./data_test/gpt_data/loss_compara_content_sentence"}
VOCAB_FILE=${24:-"./data_test/gpt_data/gpt2-vocab.json"}
MERGE_FILE=${25:-"./data_test/gpt_data/gpt2-merges.txt"}
RUN_COMMIT=${26:-"e156d2f"}

if [ $NODE_RANK -eq 0 ]; then
sed -i '/import time/a\import os' ./megatron/training.py
sed -i '/while iteration < args.train_iters:/a\        if iteration == 101: \
            cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
            os.system(cmd)' ./megatron/training.py
sed -i "/elapsed_time_per_iteration \* 1000.0)/a\        log_string += ' tpt: {:.1f} samples/s |'.format(batch_size / elapsed_time_per_iteration)" ./megatron/training.py
fi


GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

RUN_TYPE="eager"
if $GRAPH_ENABLED; then
    RUN_TYPE="graph"
fi

DP=`expr $NNODES \* $GPUS_PER_NODE \/ $MP \/ $PP`
GLOBAL_BATCH_SIZE=$((ACC * DP * MICRO_BATCH_SIZE))

# const 
TRAIN_EPOCH=0
LOAD_WEIGHT=""

hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
bias_dropout_fusion=false
save_checkpoint_period=1000


LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/$HOSTNAME/${NNODES}n${GPUS_PER_NODE}g

LOG_FILENAME=${TRAN_MODEL}_${RUN_TYPE}_nl${NUM_LAYER}_nah${NUM_ATT_HEADS}_hs${HIDDEN_SIZE}_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_DP${DP}_MP${MP}_PP${PP}_zero${ZERO_ENABLE}_stage${ZERO_STAGE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_acc${ACC}_${NNODES}n${GPUS_PER_NODE}g


if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# nsys
#nsys profile --stats true --output ${LOG_FILENAME} \
CMD="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size $MP \
        --pipeline-model-parallel-size $PP \
        --num-layers $NUM_LAYER \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATT_HEADS \
        --ffn-hidden-size $INTERMEDIATE_SIZE \
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
        --hidden-dropout $hidden_dropout_prob \
        --attention-dropout $attention_probs_dropout_prob \
        --log-interval $LOG_PERIOD \
        --save-interval $save_checkpoint_period \
        --eval-interval 2000 \
        --eval-iters 20"

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


if [[ $UNSET_DROPOUT = "true" ]]; then

    hidden_dropout_prob=0.0
    attention_probs_dropout_prob=0.0
    bias_dropout_fusion=false
    LOAD_WEIGHT=${LOG_FOLDER}/$LOG_FILENAME/model_final/

    CMD+="\
    --load $LOAD_WEIGHT \
    --no-bias-dropout-fusion"
fi

if [[ $SAVE_MODEL = "true" ]]; then

    SAVE_WEIGHT=${LOG_FOLDER}/$LOG_FILENAME/model_final/

    CMD+=" \
        --save $SAVE_WEIGHT "
fi

LOG_FILENAME=$LOG_FOLDER/$LOG_FILENAME


echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

if [ $NODE_RANK -eq 0 ]; then
    git checkout ./megatron/training.py
fi
