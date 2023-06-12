set -ex

CONFIG=$1
PLATFORM=${2:-"tencent"}
NNODES=${3:-1}
GPUS_PER_NODE=${4:-8}
NODE_RANK=${5:-0}
MASTER_ADDRS=${6:-"127.0.0.1"}
MASTER_PORTS=12345
MP=${7:-1}
PP=${8:-1}
GRAPH_ENABLED=${9:-true}
USE_FP16=${10:-true}
ACTIVATION_CHECKPOINT=${11:-false}
MICRO_BATCH_SIZE=${12:-4}
ACC=${13:-1}
ZERO_ENABLE=${14:-false}
ZERO_STAGE=${15:-2}
TRAIN_ITERS=${16:-220}
LOG_PERIOD=${17:-100}
NUM_LAYER=${18:-12}
NUM_ATT_HEADS=${19:-12}
HIDDEN_SIZE=${20:-768}
INTERMEDIATE_SIZE=${21:-3072}
HEAD_SIZE=${22:-64}
SAVE_MODEL=${23:-false}
UNSET_DROPOUT=${24:-false}

source env_${PLATFORM}.sh

if [ $NODE_RANK -eq 0 ]; then
    sed -i '/import time/a\import os' ./libai/engine/trainer.py
    sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 99: \
                        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                        os.system(cmd)' ./libai/engine/trainer.py
fi

ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
TRAN_MODEL=${CONFIG##*/}
TRAN_MODEL="LibAI_${TRAN_MODEL%*.py}"
#RUN_TIME=$(date "+%Y%m%d_%H%M%S%N")

AMP_OR="FP32"
if $USE_FP16; then
    AMP_OR="FP16"
fi

RUN_TYPE="eager"
if $GRAPH_ENABLED; then
    RUN_TYPE="graph"
fi

# const
TRAIN_EPOCH=0
LOAD_WEIGHT=""
EVALUATION_ENABLED=true
EVAL_ITER=20
hidden_dropout_prob=0.1
attention_probs_dropout_prob=0.1
bias_dropout_fusion=true
save_checkpoint_period=1000

DP=$(expr $NNODES \* $GPUS_PER_NODE \/ $MP \/ $PP)
GLOBAL_BATCH_SIZE=$((ACC * DP * MICRO_BATCH_SIZE))

LOG_FOLDER=test_logs/$HOSTNAME/${GPU_NAME}

LOG_FILENAME=${TRAN_MODEL}_${RUN_TYPE}_nl${NUM_LAYER}_nah${NUM_ATT_HEADS}_hs${HIDDEN_SIZE}_${AMP_OR}_ac${ACTIVATION_CHECKPOINT}_DP${DP}_MP${MP}_PP${PP}_zero${ZERO_ENABLE}_stage${ZERO_STAGE}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_acc${ACC}_${NNODES}n${GPUS_PER_NODE}g

if [[ $UNSET_DROPOUT = "true" ]]; then
    #sed -i 's/persistent_workers=True/#persistent_workers=True/g' ./libai/data/build.py
    sed -i 's/shuffle=True/shuffle=False/g' ./libai/data/build.py
    hidden_dropout_prob=0.0
    attention_probs_dropout_prob=0.0
    bias_dropout_fusion=false
    LOAD_WEIGHT=${LOG_FOLDER}/$LOG_FILENAME/model_final/
fi

if [[ $SAVE_MODEL = "false" ]]; then
    #sed -i 's/hooks.PeriodicCheckpointer/#&/' ./libai/engine/default.py
    if [ $NODE_RANK -eq 0 ]; then
        sed -i '/if self.cfg.train.evaluation.enabled:/i\        ret.pop()' ./libai/engine/default.py
    fi
    LOG_FOLDER=$LOG_FOLDER/${ONEFLOW_COMMIT}
fi

LOG_FILENAME=$LOG_FOLDER/$LOG_FILENAME

mkdir -p $LOG_FILENAME
echo LOG_FILENAME=$LOG_FILENAME

# nsys -delay=500
# export ONEFLOW_DEBUG_MODE=1
# export GLOG_v=3
# export ONEFLOW_PROFILER_KERNEL_PROFILE_KERNEL_FORWARD_RANGE=True

# nsys profile --stats true --output ${LOG_FILENAME} --sample none --cpuctxsw none \
python3 -m oneflow.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDRS --master_port $MASTER_PORTS \
    tools/train_net.py \
    --config-file $CONFIG \
    model.cfg.hidden_dropout_prob=$hidden_dropout_prob \
    model.cfg.attention_probs_dropout_prob=$attention_probs_dropout_prob \
    model.cfg.bias_dropout_fusion=$bias_dropout_fusion \
    model.cfg.hidden_layers=$NUM_LAYER \
    model.cfg.hidden_size=$HIDDEN_SIZE \
    model.cfg.num_attention_heads=$NUM_ATT_HEADS \
    model.cfg.intermediate_size=$INTERMEDIATE_SIZE \
    model.cfg.ffn_hidden_size=$INTERMEDIATE_SIZE \
    model.cfg.head_size=$HEAD_SIZE \
    graph.enabled=$GRAPH_ENABLED \
    train.dist.pipeline_num_layers=$NUM_LAYER \
    train.train_micro_batch_size=$MICRO_BATCH_SIZE \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.dist.tensor_parallel_size=$MP \
    train.dist.pipeline_parallel_size=$PP \
    train.amp.enabled=$USE_FP16 \
    train.activation_checkpoint.enabled=$ACTIVATION_CHECKPOINT \
    train.num_accumulation_steps=$ACC \
    train.evaluation.enabled=$EVALUATION_ENABLED \
    train.evaluation.eval_iter=$EVAL_ITER \
    train.train_iter=$TRAIN_ITERS \
    train.train_epoch=$TRAIN_EPOCH \
    train.log_period=$LOG_PERIOD \
    train.zero_optimization.enabled=$ZERO_ENABLE \
    train.zero_optimization.stage=$ZERO_STAGE \
    train.test_micro_batch_size=$MICRO_BATCH_SIZE \
    train.load_weight=$LOAD_WEIGHT \
    train.checkpointer.period=$save_checkpoint_period \
    train.output_dir=$LOG_FILENAME 2>&1 | tee ${LOG_FILENAME}/output.log

ONEFLOW_VERSION=$(python3 -c 'import oneflow; print(oneflow.__version__)')
ONEFLOW_LIBAI_COMMIT=$(git log --pretty=format:"%H" -n 1)
echo "oneflow-version(git_commit)=$ONEFLOW_VERSION" >>${LOG_FILENAME}/output.log
echo "oneflow-commit(git_commit)=$ONEFLOW_COMMIT" >>${LOG_FILENAME}/output.log
echo "oneflow-libai(git_commit)=$ONEFLOW_LIBAI_COMMIT" >>${LOG_FILENAME}/output.log

if [ $NODE_RANK -eq 0 ]; then
    git checkout ./libai/engine/*.py
    git checkout ./libai/data/build.py
fi
