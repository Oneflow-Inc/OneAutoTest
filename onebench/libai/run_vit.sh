set -ex

ONEFLOW_BRANCH_NAME=${1:-"master"}
LIBAI_BRANCH_NAME=${2:-"main"}

INSTALL=${3:-false}

if $INSTALL; then
  python3 -m pip uninstall -y oneflow
  python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu117
  #python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${ONEFLOW_BRANCH_NAME}/cu117/index.html
fi


if [ ! -d "./libai" ]; then
  git clone -b $LIBAI_BRANCH_NAME --depth 1 https://github.com/Oneflow-Inc/libai.git
fi

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

DATA_PATH=${DATA_PATH:-"/ssd/dataset/ImageNet/extract"}

sed -i "s+/path/to/imagenet+${DATA_PATH}+g" ./configs/vit_imagenet.py

# args: config-file nnodes nproc_per_node node_rank master_addr \
#       tensor_parallel_size pipeline_parallel_size graph.enabled amp activation_checkpoint \
#       train_micro_batch_size global_batch_size zero_optimization zero_optimization.stage \
#       train_iter log_period
#       hidden_layers num_attention_heads hidden_size intermediate_size head_size 

export CUDA_VISIBLE_DEVICES=0,1,4,5

#  1n1g         vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/vit_imagenet.py 1 1 0 127.0.0.1 1 1 true true true 128 1024 true 2 220 100

#  1n4g 模型并行        vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 4 1 true true true 128 1024 true 2 220 100

#  1n4g 流水并行        vit_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 4 true true true 128 1024 true 2 220 100

#  1n4g 数据并行        vit_imagenet_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 1 true true true 32 1024 true 2 220 100

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 1 true true true 64 1024 true 2 220 100

#  1n4g 3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 2 true true true 128 1024 true 2 220 100

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 2 true true true 64 1024 true 2 220 100



## vit  关Acc

#  1n1g         vit_imagenet_fp16_actrue_mp8_pp1_mb8_gb8_1n1g
bash tools/args_train.sh configs/vit_imagenet.py 1 1 0 127.0.0.1 1 1 true true true 256 256 true 2 220 100

#  1n4g 模型并行        vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 4 1 true true false 256 256 true 2 220 100

#  1n4g 流水并行        vit_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 4 true true true 256 256 true 2 220 100

#  1n4g 数据并行        vit_imagenet_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 1 true true false 64 256 true 2 220 100

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 1 true true true 128 256 true 2 220 100

#  1n4g 3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 2 true true true 256 256 true 2 220 100

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_train.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 2 true true true 128 256 true 2 220 100

