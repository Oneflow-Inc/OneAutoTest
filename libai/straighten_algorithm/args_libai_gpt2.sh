set -ex
# # bash tools/args_libai_gpt2.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

# volcengine.com
#export NCCL_IB_PCI_RELAXED_ORDERING=1
#export ONEFLOW_COMM_NET_IB_GID_INDEX=$NCCL_IB_GID_INDEX
#export ONEFLOW_COMM_NET_IB_HCA=$NCCL_IB_HCA

CONFIG=$1
NNODES=${2:-1}
GPUS_PER_NODE=${3:-8}
# Change for multinode config
NODE_RANK=${4:-0}
MASTER_ADDR=${5:-"127.0.0.1"}
MASTER_PORT=12345
MP=${6:-1}
PP=${7:-1}
USE_FP16=${8:-true}
ACTIVATION_CHECKPOINT=${9:-false}
MICRO_BATCH_SIZE=${10:-4}
GLOBAL_BATCH_SIZE=${11:-4}
STRAIGHTEN_ALGORITHM=${12:-false}
NUM_LAYER=${13:-24}
RUN_COMMIT=${14:-"01b1d32"}
TRAIN_ITERS=${15:-220}
LOG_PERIOD=${16:-100}

TRAN_MODEL="LibAI_gpt2"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=test_logs/${RUN_COMMIT}/${NNODES}n${GPUS_PER_NODE}g

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

if [ $STRAIGHTEN_ALGORITHM == true ]; then
    sed -i 's#self.config.enable_straighten_algorithm(1)#self.config.enable_straighten_algorithm(3)#g' ./libai/models/utils/graph_base.py
else
    sed -i 's#self.config.enable_straighten_algorithm(3)#self.config.enable_straighten_algorithm(1)#g' ./libai/models/utils/graph_base.py
fi

# log
#export CUDNN_LOGINFO_DBG=1
#export CUDNN_LOGDEST_DBG=cudnn.log
#export GLOG_v=3
#export ONEFLOW_DEBUG_MODE=1

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah16_hs1024_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_mp${MP}_pp${PP}_mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_al${STRAIGHTEN_ALGORITHM}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}
echo LOG_FILENAME=$LOG_FILENAME
mkdir -p $LOG_FILENAME

# nsys
#nsys profile --stats true --output ${LOG_FILENAME} \
python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
tools/train_net.py \
--config-file $CONFIG \
model.cfg.num_layers=$NUM_LAYER \
train.dist.pipeline_num_layers=$NUM_LAYER \
train.train_micro_batch_size=$MICRO_BATCH_SIZE \
train.global_batch_size=$GLOBAL_BATCH_SIZE \
train.dist.tensor_parallel_size=$MP \
train.dist.pipeline_parallel_size=$PP \
train.amp.enabled=$USE_FP16 \
train.activation_checkpoint.enabled=$ACTIVATION_CHECKPOINT \
train.train_iter=$TRAIN_ITERS \
train.log_period=$LOG_PERIOD \
train.output_dir=$LOG_FILENAME 2>&1 | tee ${LOG_FILENAME}/output.log

# zero
#train.zero_optimization.enabled=True \
#train.zero_optimization.stage=2 \

rm -rf $LOG_FILENAME/model_final

if [ $STRAIGHTEN_ALGORITHM == true ]; then
    sed -i 's#self.config.enable_straighten_algorithm(3)#self.config.enable_straighten_algorithm(1)#g' ./libai/models/utils/graph_base.py
fi
