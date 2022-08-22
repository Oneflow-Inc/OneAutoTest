
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
# parameters
#node_num=$(python3 ${SRC_DIR}/tools/get_host_num.py)
#pre_gpu_num=$(python3 ${SRC_DIR}/tools/get_pre_node_gpu_num.py)
#node_rank=$(python3 ${SRC_DIR}/tools/get_node_rank.py)
#master_ip=$(python3 ${SRC_DIR}/tools/get_master_ip.py)
#host_ip_list=$(python3 ${SRC_DIR}/tools/get_host_ip_list.py)
#git_commit=$(python3 ${SRC_DIR}/tools/get_whl_git_commit.py)
#echo "node_num=${node_num}"
#echo "pre_gpu_num=${pre_gpu_num}"
#echo "node_rank=${node_rank}"
#echo "master_ip=${master_ip}"
#echo "host_ip_list=${host_ip_list}"
#echo "git_commit=${git_commit}"


echo "SRC_DIR=${SRC_DIR}"

# upload to oss
chmod +x ${SRC_DIR}/oss/ossutil64

MODEL_DIR=${SRC_DIR}/scripts/libai
cd ${MODEL_DIR}

python3 -m pip install -e . --user

## BERT
#  1n1g         bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb24_gb24_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true false 24 24

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb1024_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 128 1024

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb128_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 128 128

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb24_gb24_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true false 24 24

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb128_gb1024_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 128 1024

#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb128_gb1024_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 128 1024

#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb192_gb1536_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 192 1536

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb64_gb512_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 64 512 48

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb16_gb64_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true false 16 64

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb512_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 128 512

#  1n4g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb2048_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 128 2048
