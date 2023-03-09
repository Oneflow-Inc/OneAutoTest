GPUS_PER_NODE=${1:-4}

git clone -b v2.12.12-1 --depth 1 https://github.com/NVIDIA/nccl.git
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
cd nccl/
OCCL_COMMIT=$(git log --pretty=format:"%H" -n 1)
make -j src.build
LD_LIBRARY_PATH="$(pwd)"
cd ../nccl-tests
make NCCL_HOME=${LD_LIBRARY_PATH}/build

# env
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/build/lib
# nccl env
export NCCL_PROTO=Simple
export NCCL_ALGO=Ring


# system 
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
OCCL_TESTS_COMMIT=$(git log --pretty=format:"%H" -n 1)

buffer_sizes=(64 128 256 512 1K 2K 4K 8K 16K 32K 64K 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M 256M 512M 1G)


if [ ! -d "./test_logs" ]; then
  mkdir -p ./test_logs
fi

for bytes in ${buffer_sizes[@]}
do
    ./build/all_gather_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_nccl_all_reduce_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/reduce_scatter_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_nccl_reduce_scatter_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/reduce_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_nccl_reduce_perf_${bytes}_${GPUS_PER_NODE}.log
    ./build/broadcast_perf -b ${bytes} -e ${bytes} -t ${GPUS_PER_NODE} -M 1 2>&1 | tee test_logs/${GPU_NAME}_nccl_broadcast_perf_${bytes}_${GPUS_PER_NODE}.log
done
