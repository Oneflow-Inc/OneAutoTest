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


if [ ! -d "./libai/projects/T5/data/training_data" ]; then
  cd ./libai/projects/T5
  mkdir -p ./data/
  wget -nc http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/wudao_180g_test_bert_tokenized_512_demo.zip
  unzip -n -d ./data/ wudao_180g_test_bert_tokenized_512_demo.zip
  mv ./data/wudao_180g_test_bert_tokenized_512_train ./data/training_data
  cd -
fi


wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./data_test/bert_data

cp -r ./data_test ./libai/

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./libai/tools/
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/extract_result.py -P ./libai/

cd libai

if [ $LIBAI_BRANCH_NAME != 'main' ]; then
    git remote set-branches origin $LIBAI_BRANCH_NAME
    git fetch --depth 1 origin $LIBAI_BRANCH_NAME
    git checkout $LIBAI_BRANCH_NAME
fi

python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user


## t5

#  1n1g         t5_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb4_1n1g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 1 0 127.0.0.1 1 1 true false true 4 4 false 0 220 100 24 16 1024 

#  1n1g         t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n1g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 256 false 0 220 100 24 16 1024 

#  1n1g         t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb32_1n1g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 32 false 0 220 100 24 16 1024 

#  1n4g 模型并行        t5_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb6_gb6_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 4 1 true false true 6 6 false 0 220 100 24 16 1024 

#  1n4g 模型并行        t5_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb256_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 4 1 true true true 32 256 false 0 220 100 24 16 1024 

#  1n4g 流水并行        t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb256_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 32 256 false 0 220 100 24 16 1024 

#  1n4g 流水并行        t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb48_gb384_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 48 384 false 0 220 100 24 16 1024 

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb16_gb128_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 16 128 48 false 0 220 100 24 16 1024 

#  1n4g 数据并行    t5_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb16_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 1 1 true false true 4 16 false 0 220 100 24 16 1024 

#  1n4g 数据并行    t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 32 128 false 0 220 100 24 16 1024 

#  1n4g 数据+模型并行   t5_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb512_1n4g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 4 0 127.0.0.1 2 1 true true true 32 512 false 0 220 100 24 16 1024 

#  1n8g 模型并行        t5_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb8_gb8_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 8 1 true false true 8 8 false 0 220 100 24 16 1024 

#  1n8g 模型并行        t5_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 8 1 true true true 32 256 false 0 220 100 24 16 1024 

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 16 256  false 0 220 100 48 24 16 1024 

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 24 384  false 0 220 100 48 24 16 1024 

#  1n8g 数据并行        t5_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 1 true false true 4 32 false 0 220 100 24 16 1024 

#  1n8g 数据并行        t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 256 false 0 220 100 24 16 1024 

#  1n8g 数据并行        t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 2048 false 0 220 100 24 16 1024 

#  1n8g 数据+模型并行   t5_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 1024 false 0 220 100 24 16 1024 

#  1n8g 数据+模型并行   t5_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 128 false 0 220 100 24 16 1024 

#  1n8g 数据+流水并行   t5_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_train.sh configs/t5_large_pretrain.py 1 8 0 127.0.0.1 1 4 true true true 32 512 false 0 220 100 24 16 1024 

python3 -m pip uninstall -y libai






