set -x

# install oneflow stable version
python3 -m pip uninstall -y oneflow
python3 -m pip install --find-links https://release.oneflow.info oneflow==0.9.0+cu117

#数据集
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./data_test/bert_data

# 脚本
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/extract_result.py -P ./

#每次从0开始
rm -rf ./libai

if [ ! -d "./libai" ]; then
    git clone -b main --depth 1 https://github.com/Oneflow-Inc/libai.git
fi

cp -r ./data_test ./libai/
cp ./args_train.sh ./libai/tools
cp ./extract_result.py ./libai/

cd libai

python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user
python3 -m pip install -e . --user

python3 -m oneflow --doctor
python3 -m pip list | grep libai
