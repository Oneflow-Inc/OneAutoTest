set -ex

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
  git clone -b $LIBAI_BRANCH_NAME --depth 1 https://github.com/Oneflow-Inc/libai.git
fi


if [ ! -d "./libai/data_test/gpt_data" ]; then
  mkdir -p ./libai/data_test/gpt_data
fi
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./libai/data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./libai/data_test/gpt_data

wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/megatron_script_tecent/onebench/libai/args_train.sh -P ./libai/tools/

cd libai
if [ $LIBAI_BRANCH_NAME != 'main' ]; then
    git remote set-branches origin $LIBAI_BRANCH_NAME
    git fetch --depth 1 origin $LIBAI_BRANCH_NAME
    git checkout $LIBAI_BRANCH_NAME
fi

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

