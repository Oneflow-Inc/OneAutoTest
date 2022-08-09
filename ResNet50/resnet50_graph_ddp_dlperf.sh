# /bash/bin
set -ex

# cd /workspace
# git clone https://github.com/Oneflow-Inc/OneAutoTest.git && cd OneAutoTest && mkdir scripts
# cd scripts && git clone https://github.com/Oneflow-Inc/models.git && cd models && git checkout v0.7.0_test_resnet50_with_ci
# cd /workspace/OneAutoTest
# python3 -m pip uninstall -y oneflow && python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu112 && bash resnet50_graph_ddp_dlperf.sh

RUN_COMMIT=${1:-"master"}
# dlperf  nsys
RUN_TYPE=${2:-"dlperf"}


LOOP_NUM=1
NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/path/to/nsys
    LOOP_NUM=1
fi


SRC_DIR=$(realpath $(dirname $0)/..)
# parameters
node_num=$(python3 ${SRC_DIR}/tools/get_host_num.py)
pre_gpu_num=$(python3 ${SRC_DIR}/tools/get_pre_node_gpu_num.py)
node_rank=$(python3 ${SRC_DIR}/tools/get_node_rank.py)
master_ip=$(python3 ${SRC_DIR}/tools/get_master_ip.py)
host_ip_list=$(python3 ${SRC_DIR}/tools/get_host_ip_list.py)
git_commit=$(python3 ${SRC_DIR}/tools/get_whl_git_commit.py)
echo "node_num=${node_num}"
echo "pre_gpu_num=${pre_gpu_num}"
echo "node_rank=${node_rank}"
echo "master_ip=${master_ip}"
echo "host_ip_list=${host_ip_list}"
echo "git_commit=${git_commit}"

echo "SRC_DIR=${SRC_DIR}"


MODEL_DIR=${SRC_DIR}/scripts/models/Vision/classification/image/resnet50
cd ${MODEL_DIR}

for MODEL_TYPE in "graph" "ddp"
do
  
  if [ $MODEL_TYPE == 'graph' ]; then
    # Graph or DDP  fp16  b512 gpu decode
    for (( j=0; j<$LOOP_NUM; j++ ))
    do
      bash examples/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/path 512 1 true python3 ${MODEL_TYPE} gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}
    done

    # Graph or DDP  fp32  b256 gpu decode
    for (( j=0; j<$LOOP_NUM; j++ ))
    do
      bash examples/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/path 256 1 false python3 ${MODEL_TYPE} gpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}
    done
    # Graph or DDP  fp16  b512 cpu decode
    for (( j=0; j<$LOOP_NUM; j++ ))
    do
      bash examples/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/79846248 512 1 true python3 ${MODEL_TYPE} cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}
    done
  fi

  # Graph or DDP  fp32  b256 cpu decode
  for (( j=0; j<$LOOP_NUM; j++ ))
  do
    bash examples/args_train_ddp_graph.sh ${node_num} ${pre_gpu_num} ${node_rank} ${master_ip} /dataset/79846248 256 1 false python3 ${MODEL_TYPE} cpu 100 false "${NSYS_BIN}" ${RUN_COMMIT}
  done

  # analysis result

  python3 ${SRC_DIR}/tools/extract_result.py --model-type ${MODEL_TYPE} --run-type ${RUN_TYPE} --test-commit ${git_commit} --test-log ${MODEL_DIR}/test_logs/$HOSTNAME --compare-commit ${git_commit} --url-path OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}
  
  ${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -f -r ${MODEL_DIR}/test_logs/$HOSTNAME/${node_num}n${pre_gpu_num}g  oss://oneflow-test/OneBrain/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/ResNet50-${MODEL_TYPE}/${RUN_TYPE}/${node_num}n${pre_gpu_num}g/


  rm -rf ${MODEL_DIR}/test_logs/$HOSTNAME
  echo "done"

done


rm -rf /dataset/e1a63606/onebench/log/$HOSTNAME

