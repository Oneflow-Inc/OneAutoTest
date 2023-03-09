GPUS_PER_NODE=${1:-4}

git clone --depth 1 https://github.com/Oneflow-Inc/occl.git
git clone --depth 1 https://github.com/Panlichen/nccl-tests.git occl-tests
cd occl/
OCCL_COMMIT=$(git log --pretty=format:"%H" -n 1)
make -j48
LD_LIBRARY_PATH="$(pwd)"
cd ../occl-tests
make src_simple.build -j62 NCCL_HOME=${LD_LIBRARY_PATH}/build

# env
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/build/lib
# nccl env
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring

# occl env
export RECV_SUCCESS_FACTOR=5
export RECV_SUCCESS_THRESHOLD=10000
export TOLERANT_UNPROGRESSED_CNT=10000
export BASE_CTX_SWITCH_THRESHOLD=8000
export NUM_TRY_TASKQ_HEAD=6
export DEV_TRY_ROUND=10
export CHECK_REMAINING_SQE_INTERVAL=10000

# system 
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
OCCL_TESTS_COMMIT=$(git log --pretty=format:"%H" -n 1)


if [ ! -d "./test_logs" ]; then
  mkdir -p ./test_logs
fi


buffer_sizes=(64 128 256 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G)
for bytes in ${buffer_sizes[@]}
do
    ./build/ofccl_all_gather_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_ofccl_all_reduce_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/ofccl_reduce_scatter_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_ofccl_reduce_scatter_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/ofccl_reduce_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_ofccl_reduce_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/ofccl_broadcast_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_ofccl_broadcast_perf_${bytes}_${GPUS_PER_NODE}.log
done
