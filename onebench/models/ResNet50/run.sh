set -ex

ONEFLOW_BRANCH_NAME=${1:-"master"}
INSTALL=${2:-false}

if $INSTALL; then
  python3 -m pip uninstall -y oneflow
  python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu112
fi


if [ ! -d "./models" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/models.git
fi

DATA_PATH=${DATA_PATH:-"/ssd/dataset/ImageNet/ofrecord"}
SRC_DIR=$(realpath $(dirname $0))
echo "SRC_DIR=${SRC_DIR}"

MODEL_DIR=${SRC_DIR}/models/Vision/classification/image/resnet50

cd ${MODEL_DIR}

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/models/ResNet50/args_train_ddp_graph.sh -P ${MODEL_DIR}/examples

# NVIDIA GeForce RTX 3080 Ti
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 96 50 1 ddp cpu false 1 100 $DATA_PATH python3
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 96 50 1 ddp cpu false 1 100 $DATA_PATH python3

bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 160 50 1 graph gpu true 1 100 $DATA_PATH python3
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 160 50 1 graph gpu true 1 100 $DATA_PATH python3

bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 40 20 4 graph gpu true 1 100 $DATA_PATH python3
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 40 20 4 graph gpu true 1 100 $DATA_PATH python3

# train
# bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 160 50 1 graph gpu true 50 100 $DATA_PATH python3

# bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 40 20 4 graph gpu true 50 100 $DATA_PATH python3
