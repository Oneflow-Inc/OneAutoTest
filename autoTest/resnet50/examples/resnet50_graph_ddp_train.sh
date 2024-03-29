
# /bash/bin
set -ex

DATA_PATH=${1:-"/ssd/dataset/ImageNet/ofrecord"}
RUN_COMMIT=${2:-"master"}
RUN_TYPE=${3:-"train"}

LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/2020.5.1/bin/nsys
    LOOP_NUM=1
fi


SRC_DIR=$(realpath $(dirname $0)/..)
echo "SRC_DIR=${SRC_DIR}"

git_commit=$(python3 ${SRC_DIR}/../../tools/get_whl_git_commit.py)
echo "git_commit=${git_commit}"

MODEL_DIR=${SRC_DIR}/scripts/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# graph fp16 gb1280
# ResNet50_graph_train_gpudecode_FP16_b160_1n8g
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 160 50 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# ResNet50_graph_train_gpudecode_FP16_b40_acc4_1n8g
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 40 50 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT} 4 20

python3 ${SRC_DIR}/../ResNet50/tools/extract_result.py --model-type "graph" --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-graph/${RUN_TYPE}

#sleep 130s

# ddp fp32 b192
# ResNet50_ddp_train_cpudecode_FP32_b192_1n4g
#bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 160 50 false python3 ddp cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

#python3 ${SRC_DIR}/../ResNet50/tools/extract_result.py --model-type "ddp" --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-ddp/${RUN_TYPE}
