PLATFORM=${1:-"tencent"}

git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM

git checkout e156d2fea7fc5c98e645f7742eb86b643956d840

if [ ! -d "./data_test/gpt_data" ]; then
    mkdir -p ./data_test/gpt_data
fi

wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./data_test/gpt_data

wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/main/onebench/libai/platform/megatron_arg_train_platform.sh -P ./examples
wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/main/onebench/libai/platform/env_${PLATFORM}.sh -P ./examples

# config platform NNODES GPUS_PER_NODE NODE_RANK MASTER_ADDR mp pp GRAPH_ENABLED USE_FP16 ACTIVATION_CHECKPOINT MICRO_BATCH_SIZE ACC ZERO_ENABLE ZERO_STAGE TRAIN_ITERS LOG_PERIOD NUM_LAYER NUM_ATT_HEADS HIDDEN_SIZE INTERMEDIATE_SIZE HEAD_SIZE SAVE_MODEL UNSET_DROPOUT

# Data Parallel
bash examples/megatron_arg_train_platform.sh ${PLATFORM} 1 8 0 127.0.0.1 1 1 true true true 2 1 false 2 220 100 48 144 2304 9216
