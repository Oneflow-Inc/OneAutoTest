
# /bash/bin
set -ex


RUN_COMMIT=${1:-"master"}
# dlperf  nsys
RUN_TYPE=${2:-"dlperf"}


NSYS_BIN=""

if [ $RUN_TYPE == 'nsys' ]; then
    NSYS_BIN=/opt/nvidia/nsight-systems/2020.5.1/bin/nsys
fi


SRC_DIR=$(realpath $(dirname $0)/..)


echo "SRC_DIR=${SRC_DIR}"

# upload to oss
chmod +x ${SRC_DIR}/oss/ossutil64

MODEL_DIR=${SRC_DIR}/scripts/models/RecommenderSystems/wide_and_deep
cd ${MODEL_DIR}



# 1n4g

# WDL_ddp_dlperf_vs2322444_devs16_hun2_b512_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 512 1100 python3 ddp

# WDL_ddp_dlperf_vs2322444_devs16_hun2_b65536_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 65536 1100 python3 ddp

# WDL_ddp_dlperf_vs2322444_devs16_hun2_b131072_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 131072 1100 python3 ddp

# WDL_graph_dlperf_vs2322444_devs16_hun2_b512_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 512 1100 python3 graph

# WDL_graph_dlperf_vs2322444_devs16_hun2_b65536_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 65536 1100 python3 graph

# WDL_graph_dlperf_vs2322444_devs16_hun2_b131072_1n4g
bash wdl_graph_ddp.sh 1 4 0 127.0.0.1 /data/dataset/wdl_ofrecord 131072 1100 python3 graph
