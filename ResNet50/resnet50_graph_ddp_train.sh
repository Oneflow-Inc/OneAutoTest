
RUN_COMMIT=${1:-"master"}
# dlperf  nsys
#RUN_TYPE=${2:-"dlperf"}
RUN_TYPE="train"

LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/2020.5.1/bin/nsys
    LOOP_NUM=1
fi
#先clone OneAutoTest的dev_resnet50_test分支仓库和models仓库

SRC_DIR=$(realpath $(dirname $0)/)
# parameters
node_num=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_host_num.py)
pre_gpu_num=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_pre_node_gpu_num.py)
node_rank=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_node_rank.py)
master_ip=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_master_ip.py)
host_ip_list=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_host_ip_list.py)
git_commit=$(python3 ${SRC_DIR}/OneAutoTest/tools/get_whl_git_commit.py)
echo "node_num=${node_num}"
echo "pre_gpu_num=${pre_gpu_num}"
echo "node_rank=${node_rank}"
echo "master_ip=${master_ip}"
echo "host_ip_list=${host_ip_list}"
echo "git_commit=${git_commit}"


echo "SRC_DIR=${SRC_DIR}"

# upload to oss
chmod +x ${SRC_DIR}/path/to/ossutil64


#MODEL_DIR=${SRC_DIR}/scripts/OneFlow-Benchmark/Classification/cnns
#cd ${MODEL_DIR}


#MODEL_TYPE="lazy"

#bash args_train.sh ${node_num} ${pre_gpu_num} 192 true 50 100 /dataset/79846248/train /dataset/79846248/validation python3 ${host_ip_list} false "${NSYS_BIN}" ${RUN_COMMIT}

#python3 ${SRC_DIR}/OneAutoTest/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

#${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/

#rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME

#sleep 130s

MODEL_DIR=${SRC_DIR}/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}
#如果运行中提示train.py: error: unrecognized arguments: , 需要把models的分支test_resnet50_with_ci中models/Vision/classification/image/resnet50/config.py中的缺少的参数代码复制到main分支中。
MODEL_TYPE="graph"

# graph fp16 b192
bash /OneAutoTest/ResNet50/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/79846248 192 50 true python3 ${MODEL_TYPE} gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}


python3 ${SRC_DIR}/OneAutoTest/ResNet50/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/.ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/

rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME

sleep 130s

# ddp fp32 b192
MODEL_TYPE="ddp"

bash /OneAutoTest/ResNet50/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/79846248 192 50 false python3 ${MODEL_TYPE} cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}


python3 ${SRC_DIR}/OneAutoTest/ResNet50/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}

${SRC_DIR}/path/to/ossutil64 -c ${SRC_DIR}/path/to/.ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/

rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME

rm -rf ${SRC_DIR}/log/$HOSTNAME

