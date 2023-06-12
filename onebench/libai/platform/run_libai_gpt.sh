set -x

PLATFORM=${1:-"tencent"}
ONEFLOW_BRANCH_NAME=${2:-"master"}
LIBAI_BRANCH_NAME=${3:-"main"}
INSTALL=${4:-true}

git config --global http.proxy http://${fast_proxy}
git config --global https.proxy https://${fast_proxy}
export http_proxy=${fast_proxy}
export https_proxy=${fast_proxy}

if $INSTALL; then
    python3 -m pip uninstall -y oneflow
    if [ $ONEFLOW_BRANCH_NAME == 'master' ]; then
        python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu117
    else
        python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_BRANCH_NAME}/cu117/index.html
    fi
fi

if [ ! -d "./libai" ]; then
    git clone -b $LIBAI_BRANCH_NAME https://github.com/Oneflow-Inc/libai.git
fi

if [ ! -d "./libai/data_test/gpt_data" ]; then
    mkdir -p ./libai/data_test/gpt_data
fi
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./libai/data_test/gpt_data

wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/main/onebench/libai/platform/arg_train_platform.sh -P ./libai/tools/
wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/main/onebench/libai/platform/env_${PLATFORM}.sh -P ./libai/tools/

cd libai

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

# config platform NNODES GPUS_PER_NODE NODE_RANK MASTER_ADDR mp pp GRAPH_ENABLED USE_FP16 ACTIVATION_CHECKPOINT MICRO_BATCH_SIZE ACC ZERO_ENABLE ZERO_STAGE TRAIN_ITERS LOG_PERIOD NUM_LAYER NUM_ATT_HEADS HIDDEN_SIZE INTERMEDIATE_SIZE HEAD_SIZE SAVE_MODEL UNSET_DROPOUT

# Data Parallel
bash tools/args_train_platform.sh configs/gpt2_pretrain.py ${PLATFORM} 1 8 0 127.0.0.1 1 1 true true true 2 1 false 2 220 100 48 144 2304 9216
