set -ex
source /home/ouyangyu/miniconda3/etc/profile.d/conda.sh

conda activate py38


ONEFLOW_BRANCH_NAME=${1:-"master"}
LIBAI_BRANCH_NAME=${2:-"main"}
INSTALL=${3:-true}

if $INSTALL; then
  python3 -m pip uninstall -y oneflow
  if [ $ONEFLOW_BRANCH_NAME == 'master' ]; then
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu117
  else
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_BRANCH_NAME}/cu117/index.html
  fi
fi

#数据集
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt  -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./data_test/bert_data

# 脚本
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/extract_result.py -P ./

#每次都重新 clone
rm -rf ./libai
if [ ! -d "./libai" ]; then
  git clone -b $LIBAI_BRANCH_NAME --depth 1 https://github.com/Oneflow-Inc/libai.git
fi
cp -r ./data_test ./libai/
cp ./args_train.sh ./libai/tools
cp ./extract_result.py ./libai/


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

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb64_1n1g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 8 64 false 0 220 100 24 16 1024 4096

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 8 8 false 0 220 100 24 16 1024 4096

#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb8_gb64_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 4 1 true true true 8 64 false 0 220 100 48 16 1024 4096

#  1n4g 流水并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb12_gb96_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 12 96 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb32_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 8 32 true 2 220 100 24 16 1024 4096

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb32_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 8 256 true 2 220 100 24 16 1024 4096


#  1n8g 模型并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb8_gb64_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 8 1 true true true 8 64 false 0 220 100 24 16 1024 4096

#  1n8g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb6_gb96_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 6 96 false 0 220 100 48 16 1024 4096

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb64_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 8 64 true 2 220 100 24 16 1024 4096

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb512_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 8 512 true 2 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb256_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 8 256 true 2 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb32_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 8 32 true 2 220 100 24 16 1024 4096

#  1n8g 数据+流水并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb8_gb128_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 4 true true true 8 128 true 2 220 100 24 16 1024 4096


#  1n8g 数据+流水并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb8_gb128_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 2 2 true true true 8 128 true 2 220 100 24 16 1024 4096


## BERT
# 3080ti  

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 16 128 false 0 220 100 24 16 1024 4096

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb32_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 32 32 false 0 220 100 24 16 1024 4096

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb256_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 4 1 true true true 32 256 false 0 220 100 48 16 1024 4096

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 32 128 true 2 220 100 24 16 1024 4096

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb16_gb128_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 4 true true true 16 128 false 0 220 100 48 16 1024 4096


#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 8 1 true true true 32 256 false 0 220 100 24 16 1024 4096

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 8 true true true 24 384 false 0 220 100 48 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 256 true 2 220 100 24 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 32 2048 true 2 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 1024 true 2 220 100 24 16 1024 4096

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 1 true true true 32 128 true 2 220 100 24 16 1024 4096

#  1n8g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 4 true true true 32 512 true 2 220 100 24 16 1024 4096

#  1n8g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 2 true true true 32 512 true 2 220 100 24 16 1024 4096

# swin 
DATA_PATH=${DATA_PATH:-"/ssd/dataset/ImageNet/extract"}

sed -i "s+/path/to/imagenet+${DATA_PATH}+g" ./configs/swin_imagenet.py


#  1n1g         vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/swin_imagenet.py 1 1 0 127.0.0.1 1 1 true true true 128 1024 true 2 220 100

#  1n4g 模型并行        vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 4 1 true true true 128 1024 true 2 220 100

#  1n4g 流水并行        vit_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 4 true true true 128 1024 true 2 220 100

#  1n4g 数据并行        vit_imagenet_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 1 true true true 32 1024 true 2 220 100

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 1 true true true 64 1024 true 2 220 100

#  1n4g 3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 2 true true true 128 1024 true 2 220 100

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 2 true true true 64 1024 true 2 220 100



##  关Acc

#  1n1g         vit_imagenet_fp16_actrue_mp8_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/swin_imagenet.py 1 1 0 127.0.0.1 1 1 true true true 256 256 true 2 220 100

#  1n4g 模型并行        vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 4 1 true true true 256 256 true 2 220 100

#  1n4g 流水并行        vit_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 4 true true true 256 256 true 2 220 100

#  1n4g 数据并行        vit_imagenet_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 1 true true true 64 256 true 2 220 100

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 1 true true true 128 256 true 2 220 100

#  1n4g 3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 2 true true true 256 256 true 2 220 100

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 2 true true true 128 256 true 2 220 100

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 1 true true true 64 2048 true 2 220 100

#  3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 2 true true true 128 2048 true 2 220 100

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 1 2 true true true 64 2048 true 2 220 100


#   数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 1 true true true 128 512 true 2 220 100

#   3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 2 true true true 256 512 true 2 220 100

#   数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 1 2 true true true 128 512 true 2 220 100


python3 -m pip uninstall -y libai

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
ONEFLOW_MODELS_COMMIT=$(git log --pretty=format:"%H" -n 1)
mv ./test_logs/$HOSTNAME/ ./master

python3 extract_result.py --test_commits $ONEFLOW_COMMIT --test_logs master/$GPU_NAME --models_commit $ONEFLOW_MODELS_COMMIT

/home/ouyangyu/ossutil64 -c /home/ouyangyu/.ossutilconfig cp -f -r master  oss://oneflow-test/OneAutoTest/onebench/libai/master
RUN_TIME=$(date "+%Y%m%d")
/home/ouyangyu/ossutil64 -c /home/ouyangyu/.ossutilconfig cp -f -r extract_result.md  oss://oneflow-test/OneAutoTest/onebench/libai/master/${RUN_TIME}_result.md

