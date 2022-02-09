# /bash/bin
# git clone https://github.com/Oneflow-Inc/models.git
# cp examples/wdl_graph_ddp.sh models/RecommenderSystems/wide_and_deep/wdl_graph_ddp.sh
# cd models/RecommenderSystems/wide_and_deep && bash wld_graph_ddp.sh

set -ex

NUM_NODES=${1:-1}
DEVICE_NUM_PER_NODE=${2:-8}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}
OFRECORD_PATH=${5:-"/dataset/f9f659c5"}
TRAIN_BATCH_SIZE=${6:-1024}
MAX_ITER=${7:-1100}
PYTHON_BIN=${8:-"python3"}
RUN_TYPE=${9:-"graph"} # graph+fp16
PRINT_INTERVAL=${10:-100}
NSYS_BIN=${11:-""}
RUN_COMMIT=${12:-"master"}

SRC_DIR=$(realpath $(dirname $0))

OFRECORD_PART_NUM=256
VOCAB_SIZE=2322444
DEEP_EMBEDDING_VEC_SIZE=16
HIDDEN_UNITS_NUM=2

TRAN_MODEL="WDL"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=${SRC_DIR}/test_logs/$HOSTNAME/${NUM_NODES}n${DEVICE_NUM_PER_NODE}g
mkdir -p $LOG_FOLDER
LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_${RUN_TYPE}_vs${VOCAB_SIZE}_devs${DEEP_EMBEDDING_VEC_SIZE}_hun${HIDDEN_UNITS_NUM}_b${TRAIN_BATCH_SIZE}_${NUM_NODES}n${DEVICE_NUM_PER_NODE}g_${RUN_COMMIT}_${RUN_TIME}

export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED

export ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=1
export ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE=1
export ONEFLOW_KERNEL_DISABLE_BLOB_ACCESS_CHECKER=1
export ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR=1

#export ONEFLOW_STREAM_CUDA_EVENT_FLAG_BLOCKING_SYNC=true

CMD=""

if [[ ! -z "${NSYS_BIN}" ]]; then
    export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=1
    export ONEFLOW_DEBUG_MODE=1
    export NCCL_DEBUG=INFO
    CMD+="${NSYS_BIN} profile --stats true --output ${LOG_FILENAME} "
    MAX_ITER=30
fi

CMD+="${PYTHON_BIN} -m oneflow.distributed.launch "
CMD+="--nproc_per_node ${DEVICE_NUM_PER_NODE} "
CMD+="--nnodes ${NUM_NODES} "
CMD+="--node_rank ${NODE_RANK} "
CMD+="--master_addr ${MASTER_ADDR} "
CMD+="${SRC_DIR}/train.py "
CMD+="--learning_rate 0.001 "
CMD+="--batch_size ${TRAIN_BATCH_SIZE} "
CMD+="--data_dir ${OFRECORD_PATH} "
CMD+="--data_part_num ${OFRECORD_PART_NUM} "
CMD+="--data_part_name_suffix_length 5 "
CMD+="--loss_print_every_n_iter ${PRINT_INTERVAL} "
CMD+="--eval_interval 0 "
CMD+="--deep_dropout_rate 0.5 "
CMD+="--max_iter 1100 "
CMD+="--hidden_units_num ${HIDDEN_UNITS_NUM} "
CMD+="--hidden_size 1024 "
CMD+="--wide_vocab_size ${VOCAB_SIZE} "
CMD+="--deep_vocab_size ${VOCAB_SIZE} "
CMD+="--deep_embedding_vec_size ${DEEP_EMBEDDING_VEC_SIZE} "
#CMD+="--use_synthetic_data "

if [ $RUN_TYPE == 'ddp' ]; then
    CMD+="--ddp "
else
    CMD+="--execution_mode ${RUN_TYPE} "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}.log"

if [[ ! -z "${NSYS_BIN}" ]]; then
    rm ${LOG_FOLDER}/*.sqlite
    mkdir -p ${LOG_FILENAME}
    rm -rf ./log/$HOSTNAME/oneflow.*
    cp ./log/$HOSTNAME/* ${LOG_FILENAME}/
fi

rm -rf ./log/$HOSTNAME
echo "done"
