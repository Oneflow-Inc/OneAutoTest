
# /bash/bin
set -ex

DATA_PATH=${1:-"/ssd/dataset/ImageNet/ofrecord"}
RUN_COMMIT=${2:-"master"}
RUN_TYPE=${3:-"dlperf"}


LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/path/to/nsys
    LOOP_NUM=1
fi


SRC_DIR=$(realpath $(dirname $0)/..)
echo "SRC_DIR=${SRC_DIR}"

git_commit=$(python3 ${SRC_DIR}/../../tools/get_whl_git_commit.py)
echo "git_commit=${git_commit}"


MODEL_DIR=${SRC_DIR}/scripts/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}


sed -i '/self.cur_batch += 1/a\\n            if self.cur_iter == 220: \
                break' ./train.py

# 1n8g

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Graph or DDP  fp32  b256 cpu decode
# ResNet50_ddp_dlperf_cpudecode_FP32_b256_1n8g
#bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 256 1 false python3 ddp cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# ResNet50_graph_dlperf_cpudecode_FP32_b256_1n8g
#bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 64 1 false python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph or DDP  fp32  b256 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP32_b256_1n8g
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 64 1 false python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP16_b512_1n8g
bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 128 1 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 cpu decode
# ResNet50_graph_dlperf_cpudecode_FP16_b512_1n8g
#bash examples/args_train_ddp_graph.sh 1 8 0 127.0.0.1 ${DATA_PATH} 128 1 true python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

sed -i '/if self.cur_iter == 220:/, +1d' ./train.py

# analysis result

python3 ${SRC_DIR}/../ResNet50/tools/extract_result.py --model-type "graph" --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-graph/${RUN_TYPE}

${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -r -f ${MODEL_DIR}/test_logs/$HOSTNAME/1n8g  oss://oneflow-test/autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-graph/${RUN_TYPE}/1n8g/


rm -rf ${MODEL_DIR}/test_logs
rm -rf ${MODEL_DIR}/log



