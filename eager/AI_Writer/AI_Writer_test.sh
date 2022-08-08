#!/usr/bin/env bash

# bash AI_Writer_test.sh d62400de800a3e5cd3e4c7a9b58fd43520826272

commit=$1

echo commit = $commit

python3 -m pip uninstall -y oneflow

wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/$commit/cu112/oneflow-0.8.1%2Bcu112.git.${commit:0:7}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

if [ $? -ne 0 ]; then
        wget https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/$commit/cu112/oneflow-0.8.1%2Bcu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
        python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
else
        python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:7}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
fi


# python3 -m pip install oneflow-0.8.1+cu112.git.${commit:0:8}-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

WORKLOADS=(true false)
TMP_COMPUTES=(true false)
STREAM_WAITS=(true false)
INFER_CACHES=(true false)

DATA_FOLDER=data
if [[ ! -z "$DATA_FOLDER" ]]; then
    mkdir -p $DATA_FOLDER
fi

# volcengine.com
unset NCCL_DEBUG

for WORKLOAD in ${WORKLOADS[@]}; do

        for TMP_COMPUTE in ${TMP_COMPUTES[@]}; do

                for STREAM_WAIT in ${STREAM_WAITS[@]}; do
                        for INFER_CACHE in ${INFER_CACHES[@]}; do
                                bash args_test_speed.sh $WORKLOAD $TMP_COMPUTE $STREAM_WAIT $INFER_CACHE 2>&1 | tee data/test_eager_commit_${WORKLOAD}_${TMP_COMPUTE}_${STREAM_WAIT}_${INFER_CACHE}_${commit:0:7}
                                break 4
                        done

                done

        done

done
