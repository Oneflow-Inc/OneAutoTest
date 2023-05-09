set -ex

# volcengine DDP env 
NNODES=$MLP_WORKER_NUM
GPUS_PER_NODE=$MLP_WORKER_GPU
NODE_RANK=$MLP_ROLE_INDEX

MASTER_ADDR=$MLP_WORKER_0_HOST
MASTER_PORT=$MLP_WORKER_0_PORT


MP=${1:-1}
PP=${2:-1}
GRAPH_ENABLED=${3:-true}
USE_FP16=${4:-true}
ACTIVATION_CHECKPOINT=${5:-false}
MICRO_BATCH_SIZE=${6:-4}
ACC=${7:-1}
ZERO_ENABLE=${8:-false}
ZERO_STAGE=${9:-2}
TRAIN_ITERS=${10:-220}
LOG_PERIOD=${11:-100}
NUM_LAYER=${12:-12}
NUM_ATT_HEADS=${13:-16}
HIDDEN_SIZE=${14:-768}
INTERMEDIATE_SIZE=${15:-3072}
HEAD_SIZE=${16:-64}

DATA_PATH=${17:-"./data_test/gpt_data/loss_compara_content_sentence"}
VOCAB_FILE=${18:-"./data_test/gpt_data/gpt2-vocab.json"}
MERGE_FILE=${19:-"./data_test/gpt_data/gpt2-merges.txt"}


SRC_DIR=$(realpath $(dirname $0)/..)
TRAN_MODEL="Megatron_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")

RUN_COMMIT=${26:-"e156d2f"}


RUN_TYPE="eager"
if $GRAPH_ENABLED; then
    RUN_TYPE="graph"
fi

# const 
TRAIN_EPOCH=0
LOAD_WEIGHT=""
EVALUATION_ENABLED=true
hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
bias_dropout_fusion=false
save_checkpoint_period=1000


LOG_FOLDER=test_logs/$HOSTNAME/${GPU_NAME}

LOG_FILENAME=${TRAN_MODEL}_${RUN_TYPE}_nl${NUM_LAYER}_nah${NUM_ATT_HEADS}_hs${HIDDEN_SIZE}_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_DP${DP}_MP${MP}_PP${PP}_zero${ZERO_ENABLE}_stage${ZERO_STAGE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_acc${ACC}_${NNODES}n${GPUS_PER_NODE}g

LOG_FILENAME=$LOG_FOLDER/$LOG_FILENAME

mkdir -p $LOG_FILENAME
echo LOG_FILENAME=$LOG_FILENAME


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

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
        --eval-interval 1000 \
        --adlr-autoresume \ 
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


if [[ $UNSET_DROPOUT = "true" ]]; then

    hidden_dropout_prob=0.0
    attention_probs_dropout_prob=0.0
    bias_dropout_fusion=true
    LOAD_WEIGHT=${LOG_FOLDER}/$LOG_FILENAME/model_final/

    CMD+="\
    --load $LOAD_WEIGHT \
    --no-bias-dropout-fusion"
fi

if [[ $SAVE_MODEL = "true" ]]; then
    #sed -i 's/hooks.PeriodicCheckpointer/#&/' ./libai/engine/default.py
    LOG_FOLDER=/${LOG_FILENAME}/${ONEFLOW_COMMIT}
    CMD+=" \
    --save $LOG_FOLDER "
fi

LOG_FILENAME=$LOG_FOLDER/$LOG_FILENAME


echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log
