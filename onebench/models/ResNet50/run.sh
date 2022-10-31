set -ex
if [ ! -d "./models" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/models.git
fi

DATA_PATH="/ssd/dataset/ImageNet/ofrecord"
SRC_DIR=$(realpath $(dirname $0))
echo "SRC_DIR=${SRC_DIR}"

MODEL_DIR=${SRC_DIR}/models/Vision/classification/image/resnet50

cd ${MODEL_DIR}/examples
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/models/ResNet50/args_train_ddp_graph.sh

cd ..

#bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 192 50 1 graph gpu true 1 100 $DATA_PATH python3

bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 192 50 1 graph gpu true 50 100 $DATA_PATH python3

bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 40 20 4 graph gpu true 50 100 $DATA_PATH python3
