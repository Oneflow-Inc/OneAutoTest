set -ex

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
ZERO_STAGE=${12:-2}
TRAIN_ITERS=${13:-220}
LOG_PERIOD=${14:-100}
NUM_LAYER=${15:-12}
NUM_ATT_HEADS=${16:-12}
HIDDEN_SIZE=${17:-768}
INTERMEDIATE_SIZE=${19:-3072}
HEAD_SIZE=${18:-64}





ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
#注释掉模型保存
sed -i 's/hooks.PeriodicCheckpointer/#&/' ./libai/engine/default.py
sed -i '/import time/a\import os' ./libai/engine/trainer.py
sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 99: \
                        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                        os.system(cmd)' ./libai/engine/trainer.py


TRAN_MODEL="LibAI_t5_mt5"
RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")
LOG_FOLDER=test_logs/${ONEFLOW_COMMIT}/${NNODES}n${GPUS_PER_NODE}g

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

# const 
TRAIN_EPOCH=0
EVALUATION_ENABLED=false


DP=`expr $NNODES \* $GPUS_PER_NODE \/ $MP \/ $PP`
ACC=`expr $GLOBAL_BATCH_SIZE \/ $DP \/ $MICRO_BATCH_SIZE`

LOG_FILENAME=$LOG_FOLDER/${TRAN_MODEL}_nl${NUM_LAYER}_nah${NUM_ATT_HEADS}_hs${HIDDEN_SIZE}_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_DP${DP}_MP${MP}_PP${PP}_zero${ZERO_ENABLE}_stage${ZERO_STAGE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_acc${ACC}_${NNODES}n${GPUS_PER_NODE}g_${RUN_TIME}
echo LOG_FILENAME=$LOG_FILENAME
mkdir -p $LOG_FILENAME


# nsys -delay=500
# export ONEFLOW_DEBUG_MODE=1
# export GLOG_v=3
# export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True

# nsys profile --stats true --output ${LOG_FILENAME} --sample none --cpuctxsw none \
python3 -m oneflow.distributed.launch \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
tools/train_net.py \
--config-file projects/T5/configs/mt5_pretrain.py \
model.cfg.hidden_layers=$NUM_LAYER \
model.cfg.hidden_size=$HIDDEN_SIZE \
model.cfg.num_attention_heads=$NUM_ATT_HEADS \
model.cfg.intermediate_size=$INTERMEDIATE_SIZE \
model.cfg.head_size=$HEAD_SIZE \
train.dist.pipeline_num_layers=$((2*NUM_LAYER)) \
train.train_micro_batch_size=$MICRO_BATCH_SIZE \
train.global_batch_size=$GLOBAL_BATCH_SIZE \
train.dist.tensor_parallel_size=$MP \
train.dist.pipeline_parallel_size=$PP \
train.amp.enabled=$USE_FP16 \
train.activation_checkpoint.enabled=$ACTIVATION_CHECKPOINT \
train.num_accumulation_steps=$ACC \
train.evaluation.enabled=$EVALUATION_ENABLED \
train.train_iter=$TRAIN_ITERS \
train.train_epoch=$TRAIN_EPOCH \
train.log_period=$LOG_PERIOD \
train.zero_optimization.enabled=$ZERO_ENABLE \
train.zero_optimization.stage=$ZERO_STAGE \
train.output_dir=$LOG_FILENAME 2>&1 | tee ${LOG_FILENAME}/output.log

git checkout ./libai/engine/*.py
