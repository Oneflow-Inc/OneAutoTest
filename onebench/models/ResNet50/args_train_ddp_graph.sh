set -ex

# bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 192 50 1 graph gpu true 1 100 /ssd/image/ python3

# machine info
NUM_NODES=${1:-1}
DEVICE_NUM_PER_NODE=${2:-8}
NODE_RANK=${3:-0}
MASTER_ADDR=${4:-"127.0.0.1"}

TRAIN_BATCH_SIZE=${5:-192}
VAL_BATCH_SIZE=${6:-50}
ACC=${7:-1}
RUN_TYPE=${8:-"ddp"} # graph+fp16
DECODE_TYPE=${9:-"cpu"}
USE_FP16=${10:-false}

EPOCH=${11:-50}
PRINT_INTERVAL=${12:-100}
# synthetic data or real data 
DATA_PATH=${13:-""}
PYTHON_BIN=${14:-"python3"}


SRC_DIR=$(realpath $(dirname $0)/..)

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')

TRAN_MODEL="resnet50"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
LOG_FOLDER=${SRC_DIR}/test_logs/$HOSTNAME/${GPU_NAME}/${ONEFLOW_COMMIT}/${NUM_NODES}n${DEVICE_NUM_PER_NODE}g

if [ ! -d $LOG_FOLDER ]; then
  mkdir -p $LOG_FOLDER
fi

synthetic="real"
if [ $DATA_PATH == '' ]; then
    synthetic="synthetic"
fi

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_${RUN_TYPE}_${synthetic}data_DC${DECODE_TYPE}_${AMP_OR}_mb${TRAIN_BATCH_SIZE}_gb$((${TRAIN_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC}))_acc${ACC}_${NUM_NODES}n${DEVICE_NUM_PER_NODE}g_${ONEFLOW_COMMIT}_${RUN_TIME}

if [ ${EPOCH} -lt 2 ];then
    sed -i '/self.cur_batch += 1/a\\n            if self.cur_iter == 320: \
                break' ${SRC_DIR}/train.py
fi
sed -i '/self.cur_batch += 1/a\\n            if self.cur_iter == 100: \
                cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                os.system(cmd)' ${SRC_DIR}/train.py


export PYTHONUNBUFFERED=1
echo PYTHONUNBUFFERED=$PYTHONUNBUFFERED

LEARNING_RATE=$(echo | awk "{print $NUM_NODES*$DEVICE_NUM_PER_NODE*$TRAIN_BATCH_SIZE*$ACC/1000}")
MOM=0.875
OFRECORD_PART_NUM=256


CMD+="${PYTHON_BIN} -m oneflow.distributed.launch "

CMD+="--nproc_per_node ${DEVICE_NUM_PER_NODE} "
CMD+="--nnodes ${NUM_NODES} "
CMD+="--node_rank ${NODE_RANK} "
CMD+="--master_addr ${MASTER_ADDR} "
CMD+="--master_port 12345 "
CMD+="${SRC_DIR}/train.py "

if [ $DATA_PATH == '' ]; then
    CMD+="--synthetic-data "
else
    CMD+="--ofrecord-path ${DATA_PATH} "
fi

CMD+="--ofrecord-part-num ${OFRECORD_PART_NUM} "
CMD+="--num-devices-per-node ${DEVICE_NUM_PER_NODE} "
CMD+="--lr ${LEARNING_RATE} "
CMD+="--momentum ${MOM} "
CMD+="--num-epochs ${EPOCH} "
CMD+="--train-batch-size ${TRAIN_BATCH_SIZE} "
CMD+="--train-global-batch-size $((${TRAIN_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC})) "
CMD+="--val-batch-size ${VAL_BATCH_SIZE} "
CMD+="--val-global-batch-size $((${VAL_BATCH_SIZE}*${NUM_NODES}*${DEVICE_NUM_PER_NODE}*${ACC})) "
CMD+="--print-interval ${PRINT_INTERVAL} "

if $USE_FP16; then
    echo USE_FP16=$USE_FP16
    CMD+="--use-fp16 --channel-last "
fi

if [ $RUN_TYPE == 'ddp' ]; then
    CMD+="--ddp "
else
    CMD+="--scale-grad --graph "
    CMD+="--fuse-bn-relu "
    CMD+="--fuse-bn-add-relu "
fi

if [ $DECODE_TYPE == 'gpu' ]; then
    CMD+="--use-gpu-decode "
fi

echo "Rum cmd ${CMD}"

$CMD 2>&1 | tee ${LOG_FILENAME}.log

echo "Writting log to ${LOG_FILENAME}.log"

echo "done"

git checkout train.py

ONEFLOW_VERSION=$(python3 -c 'import oneflow; print(oneflow.__version__)')
ONEFLOW_MODELS_COMMIT=$(git log --pretty=format:"%H" -n 1)
echo "oneflow-version(git_commit)=$ONEFLOW_VERSION" >> ${LOG_FILENAME}.log
echo "oneflow-commit(git_commit)=$ONEFLOW_COMMIT" >> ${LOG_FILENAME}.log
echo "oneflow-models(git_commit)=$ONEFLOW_MODELS_COMMIT" >> ${LOG_FILENAME}.log
