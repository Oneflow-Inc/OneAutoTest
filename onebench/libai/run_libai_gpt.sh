set -ex

git config --global http.proxy http://${fast_proxy}
git config --global https.proxy https://${fast_proxy}
export http_proxy=${fast_proxy}
export https_proxy=${fast_proxy}



ONEFLOW_BRANCH_NAME=${1:-"master"}
LIBAI_BRANCH_NAME=${2:-"main"}
INSTALL=${3:-false}

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
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./libai/data_test/gpt_data

wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/megatron_script_huoshan/onebench/libai/args_train.sh -P ./libai/tools/

cd libai

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

# mp pp GRAPH_ENABLED USE_FP16 ACTIVATION_CHECKPOINT MICRO_BATCH_SIZE ACC
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 true true true 2 1 false 2 220 100 48 144 2304 9216

bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 true true true 2 4 false 2 220 100 48 144 2304 9216
