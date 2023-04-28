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

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./libai/tools/

cd libai
if [ $LIBAI_BRANCH_NAME != 'main' ]; then
    git remote set-branches origin $LIBAI_BRANCH_NAME
    git fetch --depth 1 origin $LIBAI_BRANCH_NAME
    git checkout $LIBAI_BRANCH_NAME
fi

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user


## GPT-2
# 3080 ti

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 8 8 false 0 220 100 24 16 1024 4096


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 模型并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb2_gb2_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 8 1 true true false 2 2 false 0 220 100 24 16 1024 4096



python3 -m pip uninstall -y libai
