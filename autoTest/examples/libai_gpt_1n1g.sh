
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

MODEL_DIR=${SRC_DIR}/scripts/libai
cd ${MODEL_DIR}

python3 -m pip install -e . --user


## GPT-2

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb1_gb1_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true false 1 1

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb64_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 8 64

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb8_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 8 8

#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb2_gb2_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true false 2 2

#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb8_gb64_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 8 64

#  1n4g 流水并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb8_gb64_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 8 64

#  1n4g 流水并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb12_gb96_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 12 96

#  1n4g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb4_gb32_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 4 32 48

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb1_gb4_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true false 1 4

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb32_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 8 32

#  1n4g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb128_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 8 128
