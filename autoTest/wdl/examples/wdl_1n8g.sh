
# /bash/bin
set -ex


DATA_PATH=${1-"/data/dataset/wdl_ofrecord"}
RUN_COMMIT=${2:-"master"}
# dlperf  nsys
RUN_TYPE=${3:-"dlperf"}


NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/2020.5.1/bin/nsys
fi


SRC_DIR=$(realpath $(dirname $0)/..)
echo "SRC_DIR=${SRC_DIR}"

git_commit=$(python3 ${SRC_DIR}/../../tools/get_whl_git_commit.py)
echo "git_commit=${git_commit}"

# upload to oss
chmod +x ${SRC_DIR}/oss/ossutil64

MODEL_DIR=${SRC_DIR}/scripts/models/RecommenderSystems/wide_and_deep
cd ${MODEL_DIR}


# 1n8g

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# WDL_ddp_dlperf_vs2322444_devs16_hun2_b512_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 512 1100 python3 ddp

# WDL_ddp_dlperf_vs2322444_devs16_hun2_b65536_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 65536 1100 python3 ddp

# WDL_ddp_dlperf_vs2322444_devs16_hun2_b131072_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 131072 1100 python3 ddp


${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -r -f ${MODEL_DIR}/test_logs/$HOSTNAME/1n8g  oss://oneflow-test/autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/WDL-ddp/${RUN_TYPE}/1n8g/

rm -rf ${MODEL_DIR}/test_logs


# WDL_graph_dlperf_vs2322444_devs16_hun2_b512_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 512 1100 python3 graph

# WDL_graph_dlperf_vs2322444_devs16_hun2_b65536_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 65536 1100 python3 graph

# WDL_graph_dlperf_vs2322444_devs16_hun2_b131072_1n8g
bash wdl_graph_ddp.sh 1 8 0 127.0.0.1 ${DATA_PATH} 131072 1100 python3 graph


${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -r -f ${MODEL_DIR}/test_logs/$HOSTNAME/1n8g  oss://oneflow-test/autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/WDL-graph/${RUN_TYPE}/1n8g/

rm -rf ${MODEL_DIR}/test_logs
rm -rf ${MODEL_DIR}/log
