
# /bash/bin
set -ex

RUN_TYPE=${2:-"dlperf"}


LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/path/to/nsys
    LOOP_NUM=1
fi


SRC_DIR=$(realpath $(dirname $0)/..)

MODEL_DIR=${SRC_DIR}/scripts/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}


# 1n1g

# Graph or DDP  fp32  b256 cpu decode
# ResNet50_ddp_dlperf_cpudecode_FP32_b256_1n1g
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 ddp cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# ResNet50_graph_dlperf_cpudecode_FP32_b256_1n1g
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph or DDP  fp32  b256 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP32_b256_1n1g
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP16_b512_1n1g
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 512 1 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 cpu decode
# ResNet50_graph_dlperf_cpudecode_FP16_b512_1n1g
bash examples/args_train_ddp_graph.sh 1 1 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 512 1 true python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}



# 1n4g

# Graph or DDP  fp32  b256 cpu decode
# ResNet50_ddp_dlperf_cpudecode_FP32_b256_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 ddp cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# ResNet50_graph_dlperf_cpudecode_FP32_b256_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph or DDP  fp32  b256 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP32_b256_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 256 1 false python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 gpu decode
# ResNet50_graph_dlperf_gpudecode_FP16_b512_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 512 1 true python3 graph gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

# Graph  fp16  b512 cpu decode
# ResNet50_graph_dlperf_cpudecode_FP16_b512_1n4g
bash examples/args_train_ddp_graph.sh 1 4 0 127.0.0.1 /ssd/dataset/ImageNet/ofrecord 512 1 true python3 graph cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}

'''
# analysis result

python3 ${SRC_DIR}/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/


rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME
echo "done"


rm -rf /dataset/e1a63606/onebench/log/$HOSTNAME
'''
