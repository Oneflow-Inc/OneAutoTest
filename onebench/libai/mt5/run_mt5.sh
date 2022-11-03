set -ex

ONEFLOW_BRANCH_NAME=${1:-"master"}
LIBAI_BRANCH_NAME=${2:-"main"}

python3 -m pip uninstall -y oneflow

python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu112

# python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_BRANCH_NAME}/cu112/index.html

if [ ! -d "./libai" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/libai.git
fi


if [ ! -d "./libai/projects/T5/data/training_data" ]; then
  cd ./libai/projects/T5
  mkdir -p ./data/
  wget -nc http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/wudao_180g_test_bert_tokenized_512_demo.zip
  unzip -n -d ./data/ wudao_180g_test_bert_tokenized_512_demo.zip
  mv ./data/wudao_180g_test_bert_tokenized_512_train ./data/training_data
  cd -
fi

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./libai/tools/

cd libai
git remote set-branches origin $LIBAI_BRANCH_NAME
git fetch --depth 1 origin $LIBAI_BRANCH_NAME
git checkout $LIBAI_BRANCH_NAME

python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

# args: config-file nnodes nproc_per_node node_rank master_addr \
#       tensor_parallel_size pipeline_parallel_size graph.enabled amp activation_checkpoint \
#       train_micro_batch_size global_batch_size zero_optimization zero_optimization.stage \
#       train_iter log_period
#       hidden_layers num_attention_heads hidden_size intermediate_size head_size 

# A100
bash tools/args_train.sh projects/T5/configs/mt5_pretrain.py 1 8 0 127.0.0.1 2 1 true true 1 8 true 1 220 100 24 64 1024 32768 128
# 3080TI
# bash tools/args_train.sh projects/T5/configs/mt5_pretrain.py 1 1 0 127.0.0.1 1 1 true true 4 32 true 1 220 100 12 12 768 3072 64
