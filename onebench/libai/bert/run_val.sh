set -ex

#oneflow验证分支
ONEFLOW_VAL_BRANCH_NAME=$1
ONEFLOW_BASE_BRANCH_NAME=${2:-"master"}
LIBAI_BRANCH_NAME=${3:-"main"}

python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/$ONEFLOW_BASE_BRANCH_NAME/cu112

if [ ! -d "./libai" ]; then
  git clone -b $LIBAI_BRANCH_NAME --depth 1 https://github.com/Oneflow-Inc/libai.git
fi

if [ ! -d "./libai/data_test/bert_data" ]; then
  mkdir -p ./libai/data_test/bert_data
fi
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  -P ./libai/data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./libai/data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./libai/data_test/bert_data

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./libai/tools/

cd libai
if [ $LIBAI_BRANCH_NAME != 'main' ]; then
    git remote set-branches origin $LIBAI_BRANCH_NAME
    git fetch --depth 1 origin $LIBAI_BRANCH_NAME
    git checkout $LIBAI_BRANCH_NAME
fi


#python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

# args: config-file nnodes nproc_per_node node_rank master_addr \
#       tensor_parallel_size pipeline_parallel_size graph.enabled amp activation_checkpoint \
#       train_micro_batch_size global_batch_size zero_optimization zero_optimization.stage \
#       train_iter log_period
#       hidden_layers num_attention_heads hidden_size intermediate_size head_size 

bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 128 false 0 1 1 24 16 1024 4096 64 true false

bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 128 false 0 220 1 24 16 1024 4096 64 false true

python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_VAL_BRANCH_NAME}/cu112/index.html

bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 128 false 0 220 1 24 16 1024 4096 64 false true



