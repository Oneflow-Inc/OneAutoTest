
git clone https://github.com/NVIDIA/Megatron-LM.git

cd Megatron-LM

git checkout e156d2fea7fc5c98e645f7742eb86b643956d840


if [ ! -d "./Megatron-LM/data_test/gpt_data" ]; then
    mkdir -p ./Megatron-LM/data_test/gpt_data
fi

wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./data_test/gpt_data


wget https://github.com/Tendo33/OneAutoTest/raw/megatron_script/onebench/libai/megatron_args_pretrain_gpt2.sh -P ./examples


#1n1g
bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 1 1 true true true 5 5 false 2 220 1 24 16 768 3072 64 true false

#1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 8 1 true true true 32 256 false 2 220 1 24 16 768 3072 64 true false