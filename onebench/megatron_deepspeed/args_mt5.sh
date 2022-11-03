
set -ex
# A100
# bash args_deepspeed_t5.sh 1 8 0 127.0.0.1 2 1 true true 1 8 true 1 220 100 24 64 1024 32768 128
# 3080TI
# bash args_deepspeed_t5.sh 1 8 0 127.0.0.1 2 1 true true 1 8 true 1 220 100 12 12 768 3072 64


export OMP_NUM_THREADS=1


NNODES=${1:-1}
GPUS_PER_NODE=${2:-8}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
MASTER_PORT=12345
MP=${5:-1}
PP=${6:-1}
USE_FP16=${7:-true}
ACTIVATION_CHECKPOINT=${8:-false}
MICRO_BATCH_SIZE=${9:-4}
GLOBAL_BATCH_SIZE=${10:-4}
ZERO_ENABLE=${11:-false}
ZERO_STAGE=${12:-1}
TRAIN_ITERS=${13:-220}
LOG_PERIOD=${14:-100}
NUM_LAYER=${15:-12}
NUM_ATT_HEADS=${16:-12}
HIDDEN_SIZE=${17:-768}
INTERMEDIATE_SIZE=${18:-32768}
HEAD_SIZE=${19:-64}


VOCAB_FILE=libai_dataset/bert-base-chinese-vocab.txt
DATA_PATH=libai_dataset/loss_compara_content_sentence


DP=`expr $NNODES \* $GPUS_PER_NODE \/ $MP \/ $PP`
ACC=`expr $GLOBAL_BATCH_SIZE \/ $DP \/ $MICRO_BATCH_SIZE`


SRC_DIR=$(realpath $(dirname $0)/)
TRAN_MODEL="Megatron-Deepspeed_t5"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$RUN_COMMIT/${NNODES}n${GPUS_PER_NODE}g
if [[ ! -z "$LOG_FOLDER" ]]; then
    mkdir -p $LOG_FOLDER
fi

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi


sed -i 's/assert mask is None/#&/' ./megatron/model/fused_softmax.py
sed -i "/.format(learning_rate)/a\        log_string += ' tpt: {:.1f} samples/s |'.format(batch_size / elapsed_time_per_iteration)" ./megatron/training.py
sed -i '/import time/a\import os' ./megatron/training.py
sed -i '/while iteration < args.train_iters:/a\        if iteration == 101: \
            cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
            os.system(cmd)' ./megatron/training.py


LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah${NUM_ATT_HEADS}_hs${HIDDEN_SIZE}_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_DP${DP}_MP${MP}_PP${PP}_zero${ZERO_ENABLE}_stage${ZERO_STAGE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_acc${ACC}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

T5_ARGS="--num-layers $NUM_LAYER \
         --hidden-size $HIDDEN_SIZE \
         --num-attention-heads $NUM_ATT_HEADS \
         --kv-channels 128 \
         --ffn-hidden-size $INTERMEDIATE_SIZE \
         --encoder-seq-length 512 \
         --decoder-seq-length 128 \
         --max-position-embeddings 512 \
         --lr 0.0001 \
         --lr-decay-iters 990000 \
         --train-iters $TRAIN_ITERS \
         --min-lr 0.00001 \
         --lr-warmup-fraction 0.01 \
         --micro-batch-size $MICRO_BATCH_SIZE \
         --global-batch-size $GLOBAL_BATCH_SIZE \
         --vocab-file $VOCAB_FILE \
         --vocab-extra-ids 100 \
         --split 949,50,1 \
         "

OUTPUT_ARGS=" \
    --log-interval $LOG_PERIOD \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    "
# --checkpoint-activations \

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    "
# --load $CHECKPOINT_PATH \


config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
    "zero_optimization": {
        "stage": $ZERO_STAGE
    },
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "steps_per_print": 2000,
    "gradient_clipping": 1,
    "train_batch_size": $GLOBAL_BATCH_SIZE,
    "zero_allow_untested_optimizer": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "
export LOGLEVEL=WARNING
# LAUNCHER="deepspeed -num_gpus $GPUS_PER_NODE"

#nsys profile --stats true --output ${LOG_FILENAME} --sample none --cpuctxsw none \
CMD="python3 -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_t5.py \
    $T5_ARGS \
    $OUTPUT_ARGS \
    $DATA_ARGS \
    --tensor-model-parallel-size $MP \
    --pipeline-model-parallel-size $PP \
    --DDP-impl local "

if $ZERO_ENABLE; then
    CMD+=" \
      $DEEPSPEED_ARGS "
fi

if $USE_FP16; then
    CMD+=" \
      --fp16 "
fi

if $ACTIVATION_CHECKPOINT; then
    CMD+=" \
      --checkpoint-activations "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

git checkout megatron/model/fused_softmax.py
git checkout megatron/training.py

