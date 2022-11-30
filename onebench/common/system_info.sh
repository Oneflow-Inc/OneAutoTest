#!/bin/bash

CPU_NAME="$(lscpu | grep "Model name:" | sed -r 's/Model name:\s{1,}//g')"
CPU_CORE="$(lscpu | grep "CPU(s):" | sed -r 's/CPU(s):\s{1,}//g')"
echo $CPU_CORE
CPU_MEM="$(free -h | grep Mem: | awk '{ print $2 }')"

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

GPU_MEM="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader)"
GPU_MEM="${GPU_MEM// /_}"
GPU_RAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{ printf "%.0f\n", $1 / 1024 }' | head -n1)

NVIDIA_DRIVER="$(nvidia-smi | grep "Driver Version:" | awk '{ print $3 }')"

CUDA_VERSION="$(nvcc --version | grep release | awk '{ print $NF }')"

CUDNN_MAJOR="$(cat /usr/local/cudnn/include/cudnn.h | grep "#define CUDNN_MAJOR" | awk '{ print $NF }')"
CUDNN_MINOR="$(cat /usr/local/cudnn/include/cudnn.h | grep "#define CUDNN_MINOR" | awk '{ print $NF }')"
CUDNN_PATCHLEVEL="$(cat /usr/local/cudnn/include/cudnn.h | grep "#define CUDNN_PATCHLEVEL" | awk '{ print $NF }')"
CUDNN_VERSION=${CUDNN_MAJOR}"."${CUDNN_MINOR}"."${CUDNN_PATCHLEVEL}

MB="$(cat /sys/devices/virtual/dmi/id/board_{vendor,name,version} | tr '\n' ' ')"

PLATFORM_NAME="$(cat /etc/os-release | grep "PRETTY_NAME=" | cut -c 14- | rev | cut -c 2- | rev)"

OF_VERSION="$(python -c "import oneflow; print(oneflow.__version__)" | awk '{ print $NF }')"

RESULTS_PATH=./${GPU_NAME}
mkdir -p $RESULTS_PATH

SYSTEM_FILE=${RESULTS_PATH}/sys_info.txt

echo "CPU: "${CPU_NAME} >> $SYSTEM_FILE
echo "CPU Memory: "${CPU_MEM} >> $SYSTEM_FILE
echo "GPU: "${GPU_NAME} >> $SYSTEM_FILE
echo "GPU Memory: "${GPU_MEM} >> $SYSTEM_FILE
echo "NVIDIA driver: "${NVIDIA_DRIVER} >> $SYSTEM_FILE
echo "CUDA Version: "${CUDA_VERSION} >> $SYSTEM_FILE
echo "CUDNN Version: "$CUDNN_VERSION >> $SYSTEM_FILE
echo "Motherboard: "${MB} >> $SYSTEM_FILE
echo "OS: "${PLATFORM_NAME} >> $SYSTEM_FILE
echo "OneFlow Version: "${OF_VERSION} >> $SYSTEM_FILE

chmod -R a+rwx $SYSTEM_FILE