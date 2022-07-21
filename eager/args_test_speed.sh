#!/usr/bin/env bash

set -uo pipefail

ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=${1:-false}
ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=${2:-true}
ONEFLOW_VM_ENABLE_STREAM_WAIT=${3:-true}
ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=${4:-true}

#export ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD=$ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD
#echo ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD=$ONEFLOW_VM_WORKLOAD_ON_SCHEDULER_THREAD

export ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=$ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD
echo ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD=$ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD


export ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=$ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM
echo ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM=$ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM

export ONEFLOW_VM_ENABLE_STREAM_WAIT=$ONEFLOW_VM_ENABLE_STREAM_WAIT
echo ONEFLOW_VM_ENABLE_STREAM_WAIT=$ONEFLOW_VM_ENABLE_STREAM_WAIT

export ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=$ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE
echo ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE=$ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE

rc=0
# accumulate the score of every test
trap 'rc=$(($rc + $?))' ERR

#export ONEFLOW_MODELS_DIR=/dataset/e1a63606/onebench/scripts/models_nsys

export ONEFLOW_MODELS_DIR=/path/to/models
echo "test on master"

# rm result || true

cd $ONEFLOW_MODELS_DIR

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

for((i=1;i<=9;i++));

do

echo "i = "$(expr $i);

sleep 10s

rm result || true



# python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 100 | check_relative_speed 1.05 | check_millisecond_time 129.0 2 | write_to_file_and_print
# python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 100 | check_relative_speed 1.04 | write_to_file_and_print
# python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 200 | check_relative_speed 1.01 | write_to_file_and_print
# python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.99 | write_to_file_and_print

python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.95 | write_to_file_and_print

# python3 scripts/swin_dataloader_compare_speed_with_pytorch.py --batch_size 32 --num_workers 1 | write_to_file_and_print
# python3 scripts/swin_dataloader_compare_speed_with_pytorch.py --batch_size 32 --num_workers 4 | write_to_file_and_print
# python3 scripts/swin_dataloader_compare_speed_with_pytorch.py --batch_size 32 --num_workers 8 | write_to_file_and_print

export OMP_NUM_THREADS=1
# python3 -m oneflow.distributed.launch --master_port 36570 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 100 --ddp | check_relative_speed 1.12 | check_millisecond_time 136.3 2 | write_to_file_and_print
# python3 -m oneflow.distributed.launch --master_port 32560 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 100 --ddp | check_relative_speed 1.1 | write_to_file_and_print
# python3 -m oneflow.distributed.launch --master_port 31090 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 1.18 | write_to_file_and_print
# python3 -m oneflow.distributed.launch --master_port 33748 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 1.18 | write_to_file_and_print


python3 -m oneflow.distributed.launch --master_port 31349 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 1.15 | write_to_file_and_print

done

LOG_FILENAME_1n1d=data/resnet50_eager_1x3x224x224_ws1_${ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD}_${ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM}_${ONEFLOW_VM_ENABLE_STREAM_WAIT}_${ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE}_1n1d
mkdir -p $LOG_FILENAME_1n1d
nsys profile --stats true --output ${LOG_FILENAME_1n1d} \
python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.95 | write_to_file_and_print

LOG_FILENAME_1n2d=data/resnet50_eager_1x3x224x224_ws2_${ONEFLOW_VM_COMPUTE_ON_WORKER_THREAD}_${ONEFLOW_AD_PUT_LOSS_ON_TMP_COMPUTE_STREAM}_${ONEFLOW_VM_ENABLE_STREAM_WAIT}_${ONEFLOW_EAGER_ENABLE_LOCAL_INFER_CACHE}_1n2d
mkdir -p $LOG_FILENAME_1n2d
nsys profile --stats true --output ${LOG_FILENAME_1n2d} \
python3 -m oneflow.distributed.launch --master_port 31349 --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 1.15 | write_to_file_and_print

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

