
# /bash/bin
set -ex

RUN_COMMIT=${1:-"master"}
RUN_TYPE=${2:-"train"}

LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/2020.5.1/bin/nsys
    LOOP_NUM=1
fi


SRC_DIR=$(realpath $(dirname $0)/..)
echo "SRC_DIR=${SRC_DIR}"

# upload to oss
chmod +x ${SRC_DIR}/oss/ossutil64


MODEL_DIR=${SRC_DIR}/scripts/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}


# graph fp16 b192
# ResNet50_graph_train_gpudecode_FP16_b192_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 96 50 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

'''
python3 ${SRC_DIR}/OneAutoTest/ResNet50/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/.ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/

rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME
'''
sleep 130s

# ddp fp32 b192
# ResNet50_ddp_train_cpudecode_FP32_b192_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 128 50 false python3 ddp cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

'''
python3 ${SRC_DIR}/OneAutoTest/ResNet50/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

${SRC_DIR}/path/to/ossutil64 -c ${SRC_DIR}/path/to/.ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/

rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME

rm -rf ${SRC_DIR}/log/$HOSTNAME
'''
