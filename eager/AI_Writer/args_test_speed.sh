#!/usr/bin/env bash

set -uo pipefail

ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=${1:-true}
ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=${2:-true}
ONEFLOW_VM_ENABLE_STREAM_WAIT=${3:-true}
ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=${4:-true}
INFER_MODEL=${5:-"both"}

OUTPUT_LENGTH=${6:-20}

ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=true

export ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=$ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD
echo ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=$ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD

export ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=$ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM
echo ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=$ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM

export ONEFLOW_VM_ENABLE_STREAM_WAIT=$ONEFLOW_VM_ENABLE_STREAM_WAIT
echo ONEFLOW_VM_ENABLE_STREAM_WAIT=$ONEFLOW_VM_ENABLE_STREAM_WAIT

export ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=$ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE
echo ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=$ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE

export ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=$ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE
echo ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE=$ONEFLOW_EAGER_LOCAL_TO_GLOBAL_BALANCED_OVERRIDE

rc=0
# accumulate the score of every test
trap 'rc=$(($rc + $?))' ERR

#export ONEFLOW_MODELS_DIR=/path/to/models
#echo "test on master"

# rm result || true

#cd $ONEFLOW_MODELS_DIR

unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

function check_relative_speed {
  # Default score is 1
  SCORE=${2:-1}
  awk -F'[:(]' -v threshold=$1 -v score=$SCORE 'BEGIN { ret=2 } /Relative speed/{ if ($2 >= threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=score }} {print $0} END { exit ret }'
}

function check_millisecond_time {
  # Default score is 1
  SCORE=${2:-1}
  awk -F'[:(]' -v threshold=$1 -v score=$SCORE 'BEGIN { ret=2 } /OneFlow/{ if (substr($2, 2, length($2) - 4) <= threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=score }} { print $0 } END { exit ret }'
}

function write_to_file_and_print {
  tee -a result
  printf "\n" >> result
}

python3 -m oneflow --doctor

for((i=1;i<=11;i++));

do

echo "i = "$(expr $i);

rm result || true

if [ "$INFER_MODEL" = "oneflow" ]; then
python3 compare_speed_with_pytorch.py AI-Writer --output_preLen ${OUTPUT_LENGTH} --only-oneflow | check_relative_speed 0.95 | write_to_file_and_print

elif [ "$INFER_MODEL" = "pytorch" ]; then
python3 compare_speed_with_pytorch.py AI-Writer --output_preLen ${OUTPUT_LENGTH} --only-pytorch | check_relative_speed 0.95 | write_to_file_and_print

else
python3 compare_speed_with_pytorch.py AI-Writer --output_preLen ${OUTPUT_LENGTH} | check_relative_speed 0.95 | write_to_file_and_print

fi

export OMP_NUM_THREADS=1

#python3 -m oneflow.distributed.launch --master_port 31349 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 1.15 | write_to_file_and_print

done


if [ "$INFER_MODEL" = "oneflow" ]; then
LOG_FILENAME=/path/to/writer/data/oneflow/AI_Writer_eager_outlen${OUTPUT_LENGTH}_ws1_${ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD}_${ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM}_${ONEFLOW_VM_ENABLE_STREAM_WAIT}_${ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE}
nsys profile --stats true --output ${LOG_FILENAME} --sample none \
python3 compare_speed_with_pytorch.py AI-Writer --output_preLen ${OUTPUT_LENGTH} --only-oneflow | check_relative_speed 0.95 | write_to_file_and_print


result="GPU Name: `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` \n\n `cat result`"
# escape newline for github actions: https://github.community/t/set-output-truncates-multiline-strings/16852/2
# note that we escape \n and \r to \\n and \\r (i.e. raw string "\n" and "\r") instead of %0A and %0D,
# so that they can be correctly handled in javascript code
# result="${result//'%'/'%25'}"
# result="${result//$'\n'/'\\n'}"
# result="${result//$'\r'/'\\r'}"

# echo "::set-output name=stats::$result"

# Only fail when the sum of score >= 2
if (( $rc >= 2 ))
then
  exit 1
else
  exit 0
fi