set -ex

source /home/ouyangyu/miniconda3/etc/profile.d/conda.sh

conda activate base

ONEFLOW_BRANCH_NAME=${1:-"master"}

python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${ONEFLOW_BRANCH_NAME}/cu117

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/models/ResNet50/args_train_ddp_graph.sh -P ./
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/models/ResNet50/extract_result.py -P ./

#每次都重新clone
rm -rf models

git clone --depth 1 https://github.com/Oneflow-Inc/models.git

DATA_PATH=${DATA_PATH:-"/ssd/dataset/ImageNet/ofrecord"}
SRC_DIR=$(realpath $(dirname $0))
echo "SRC_DIR=${SRC_DIR}"

MODEL_DIR=${SRC_DIR}/models/Vision/classification/image/resnet50
cp ./args_train_ddp_graph.sh ${MODEL_DIR}/examples/
cp ./extract_result.py ${MODEL_DIR}/

cd ${MODEL_DIR}

bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 160 50 1 graph gpu true 50 100 $DATA_PATH python3

bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 40 20 4 graph gpu true 50 100 $DATA_PATH python3

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
ONEFLOW_MODELS_COMMIT=$(git log --pretty=format:"%H" -n 1)
mv ./test_logs/$HOSTNAME/ ./master
python3 extract_result.py --test_commits $ONEFLOW_COMMIT --test_logs master/$GPU_NAME --models_commit $ONEFLOW_MODELS_COMMIT

/home/ouyangyu/ossutil64 -c /home/ouyangyu/.ossutilconfig cp -f -r master  oss://oneflow-test/OneAutoTest/onebench/resnet50/master
RUN_TIME=$(date "+%Y%m%d")
/home/ouyangyu/ossutil64 -c /home/ouyangyu/.ossutilconfig cp -f -r extract_result.md  oss://oneflow-test/OneAutoTest/onebench/resnet50/master/${RUN_TIME}_result.md

