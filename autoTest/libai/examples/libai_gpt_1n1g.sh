
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
