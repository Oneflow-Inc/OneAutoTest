set -ex

# docker run --rm -it --shm-size=16g --ulimit memlock=-1 --privileged --name megatron_deepspeed_mt5 --net host -v /data/workspace:/workspace nvcr.io/nvidia/pytorch:21.07-py3


RE_BUILD=${1:-false}
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

if [ ! -d "./Megatron-DeepSpeed" ]; then
  git clone --depth 1 https://github.com/bigscience-workshop/Megatron-DeepSpeed.git
  cd Megatron-DeepSpeed
  git remote set-branches origin 09a35f53abac96903fee50787426b7ee5f63fc62
  git fetch --depth 1 origin 09a35f53abac96903fee50787426b7ee5f63fc62
  git checkout 09a35f53abac96903fee50787426b7ee5f63fc62
  cd ..
fi

if [ ! -d "./Megatron-DeepSpeed/apex" ]; then
  cd Megatron-DeepSpeed
  git clone --depth 1 https://github.com/NVIDIA/apex.git
  cd apex
  git remote set-branches origin 6a40a0ad9ff3d6ebea715cf28faf39792312acbf
  git fetch --depth 1 origin 6a40a0ad9ff3d6ebea715cf28faf39792312acbf
  git checkout 6a40a0ad9ff3d6ebea715cf28faf39792312acbf
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  cd ../../
fi

if [ ! -d "./Megatron-DeepSpeed/deepspeed" ]; then
  cd Megatron-DeepSpeed
  git clone  --depth 1 https://github.com/microsoft/deepspeed.git
  cd deepspeed
  git remote set-branches origin 3432c740e99cab9e78c67b59ca3c53d49e4b29b7
  git fetch --depth 1 origin 3432c740e99cab9e78c67b59ca3c53d49e4b29b7
  git checkout 3432c740e99cab9e78c67b59ca3c53d49e4b29b7
  rm -rf build
  pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
  cd ../../
fi

if $RE_BUILD; then
  cd Megatron-DeepSpeed/apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  cd ../deepspeed
  rm -rf build
  pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
  cd ../../
fi


pip install transformers


if [ ! -d "./Megatron-DeepSpeed/libai_dataset" ]; then
  mkdir -p ./Megatron-DeepSpeed/libai_dataset
  wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt -P ./Megatron-DeepSpeed/libai_dataset
  wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./Megatron-DeepSpeed/libai_dataset
  wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./Megatron-DeepSpeed/libai_dataset
fi

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/megatron_deepspeed/args_mt5.sh -P ./Megatron-DeepSpeed

cd Megatron-DeepSpeed

# args: nnodes nproc_per_node node_rank master_addr \
#       tensor_parallel_size pipeline_parallel_size amp activation_checkpoint \
#       train_micro_batch_size global_batch_size zero_optimization zero_optimization.stage \
#       train_iter log_period
#       hidden_layers num_attention_heads hidden_size
#       head_size intermediate_size

# A100
bash args_mt5.sh 1 8 0 127.0.0.1 2 1 true true 1 8 true 1 220 100 24 64 1024 32768 128
bash args_mt5.sh 1 8 0 127.0.0.1 2 1 true true 2 16 true 1 220 100 24 64 1024 32768 128
bash args_mt5.sh 1 8 0 127.0.0.1 2 1 true true 4 32 true 1 220 100 24 64 1024 32768 128
# 3080TI
#bash args_mt5.sh 1 1 0 127.0.0.1 2 1 true true 1 8 true 1 220 100 12 12 768 3072 64
