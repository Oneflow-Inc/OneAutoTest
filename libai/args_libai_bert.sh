set -ex
# # bash tools/args_libai_bert.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

CONFIG=$1
NNODES=${2:-1}
GPUS_PER_NODE=${3:-8}
# Change for multinode config
NODE_RANK=${4:-0}
MASTER_ADDR=${5:-"127.0.0.1"}
MASTER_PORT=12345
MP=${6:-1}
PP=${7:-1}
USE_FP16=${8:-"True"}
ACTIVATION_CHECKPOINT=${9:-"False"}
MICRO_BATCH_SIZE=${10:-4}
GLOBAL_BATCH_SIZE=${11:-4}
RUN_COMMIT=${12:-"01b1d32"}
TRAIN_ITERS=${13:-320}
LOG_PERIOD=${14:-100}

TRAN_MODEL="LibAI_bert"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=test_logs/${RUN_COMMIT}/${NNODES}n${GPUS_PER_NODE}g

AMP_OR="FP32"
if [ $USE_FP16 == 'True' ]; then
    AMP_OR="FP16"
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl24_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}
mkdir -p $LOG_FILENAME
python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
tools/train_net.py \
--config-file $CONFIG \
train.train_micro_batch_size=$MICRO_BATCH_SIZE \
train.global_batch_size=$GLOBAL_BATCH_SIZE \
train.dist.tensor_parallel_size=$MP \
train.dist.pipeline_parallel_size=$PP \
train.amp.enabled=$USE_FP16 \
train.activation_checkpoint.enabled=$ACTIVATION_CHECKPOINT \
train.train_iter=$TRAIN_ITERS \
train.log_period=$LOG_PERIOD \
train.output_dir=$LOG_FILENAME 2>&1 | tee ${LOG_FILENAME}/output.log

rm -rf $LOG_FILENAME/model_final
