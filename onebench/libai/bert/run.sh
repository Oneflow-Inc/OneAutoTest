set -ex

ONEFLOW_BRANCH_NAME=${1:-"master"}
LIBAI_BRANCH_NAME=${2:-"main"}

INSTALL=${3:-false}

if $INSTALL; then
  python3 -m pip uninstall -y oneflow
  python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu112
  #python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_BRANCH_NAME}/cu112/index.html
fi


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

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user

# args: config-file nnodes nproc_per_node node_rank master_addr \
#       tensor_parallel_size pipeline_parallel_size graph.enabled amp activation_checkpoint \
#       train_micro_batch_size global_batch_size zero_optimization zero_optimization.stage \
#       train_iter log_period
#       hidden_layers num_attention_heads hidden_size intermediate_size head_size 

# eager test
# bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 false false false 1 1 false 0 220 100 24 16 1024 4096

## BERT
# 3080ti  
#  1n1g         bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb6_gb6_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true false 4 4 false 0 220 100 24 16 1024 4096

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 16 128 false 0 220 100 24 16 1024 4096

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb32_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 32 false 0 220 100 24 16 1024 4096


export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb6_gb6_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 4 1 true true false 6 6 false 0 220 100 24 16 1024 4096

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb256_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 4 1 true true true 32 256 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb16_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 1 true true false 4 16 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 32 128 false 0 220 100 24 16 1024 4096


#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb256_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 32 256 false 0 220 100 24 16 1024 4096

#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb48_gb384_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 48 384 false 0 220 100 24 16 1024 4096

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb16_gb128_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 16 128 false 0 220 100 48 16 1024 4096


#  1n4g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb512_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 2 1 true true true 32 512 false 0 220 100 24 16 1024 4096


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb32_gb32_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 8 1 true true false 32 32 false 0 220 100 24 16 1024 4096

#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 8 1 true true true 32 256 false 0 220 100 24 16 1024 4096

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 16 256 false 0 220 100 48 16 1024 4096

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 24 384 false 0 220 100 48 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true false 4 32 false 0 220 100 24 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 256 false 0 220 100 24 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 2048 false 0 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 1024 false 0 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 128 false 0 220 100 24 16 1024 4096

#  1n8g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 4 true true true 32 512 false 0 220 100 24 16 1024 4096

python3 -m pip uninstall -y libai