

if [ ! -d "./Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git
fi

cd Megatron-LM
git checkout e156d2fea7fc5c98e645f7742eb86b643956d840


if [ ! -d "./data_test/gpt_data" ]; then
    mkdir -p ./data_test/gpt_data
fi

wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin  -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx  -P ./data_test/gpt_data


wget -nc https://github.com/Oneflow-Inc/OneAutoTest/raw/megatron_script_huoshan/onebench/libai/megatron_args_pretrain_gpt2.sh -P ./examples



bash examples/megatron_args_pretrain_gpt2.sh 1 1 true true true 2 1 false 2 220 100 48 144 2304 9216


bash examples/megatron_args_pretrain_gpt2.sh 1 1 true true true 2 4 false 2 220 100 48 144 2304 9216
