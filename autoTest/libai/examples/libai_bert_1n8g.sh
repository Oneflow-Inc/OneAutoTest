
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

git_commit=$(python3 ${SRC_DIR}/../../tools/get_whl_git_commit.py)
echo "git_commit=${git_commit}"

# upload to oss
chmod +x ${SRC_DIR}/oss/ossutil64

MODEL_DIR=${SRC_DIR}/scripts/libai
cd ${MODEL_DIR}

python3 -m pip install -e . --user


## BERT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb32_gb32_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true false 32 32

#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true true 32 256

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 16 256 48

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 24 384 48

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true false 4 32

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 32 256

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 32 2048

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 32 1024

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 32 128

#  1n8g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 4 true true 32 512


${SRC_DIR}/oss/ossutil64 -c ${SRC_DIR}/oss/ossutilconfig cp -r -f ${MODEL_DIR}/test_logs/$HOSTNAME/1n8g  oss://oneflow-test/autoTest/commit/${RUN_COMMIT}/$(date "+%Y%m%d")/${git_commit}/libai/1n8g/


rm -rf ${MODEL_DIR}/test_logs
rm -rf ${MODEL_DIR}/log

