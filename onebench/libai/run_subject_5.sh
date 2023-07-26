set -x

# install oneflow stable version
python3 -m pip uninstall -y oneflow
python3 -m pip install --find-links https://release.oneflow.info oneflow==0.9.0+cu117

# 下载数据集
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./data_test/gpt_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin -P ./data_test/bert_data
wget -nc https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx -P ./data_test/bert_data

# 下载脚本
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/args_train.sh -P ./
wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/Subject_5/onebench/libai/extract_result.py -P ./

# 每次重新 clone
rm -rf libai

git clone -b main --depth 1 https://github.com/Oneflow-Inc/libai.git

cp -r ./data_test ./libai/
cp ./args_train.sh ./libai/tools
cp ./extract_result.py ./libai/

cd libai
# 安装 libai
python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user
python3 -m pip install -e . --user

# 检查安装是否成功
python3 -m oneflow --doctor
python3 -m pip list | grep libai

## Bert + Graphe
# 3090

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb16_acc4_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 4 16 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb64_acc4_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 4 64 false 0 220 100 24 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb128_acc4_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 4 128 false 0 220 100 24 16 1024 4096

## Bert + Eager
# 3090

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb16_acc4_1n1g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 1 0 127.0.0.1 1 1 false true true 4 16 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb64_acc4_1n4g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 4 0 127.0.0.1 1 1 false true true 4 64 false 0 220 100 24 16 1024 4096

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb128_acc4_1n8g
bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 1 1 false true true 4 128 false 0 220 100 24 16 1024 4096

## GPT-2 + Graphe
# 3090

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb8_acc4_1n1g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 0 127.0.0.1 1 1 true true true 2 8 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb32_acc4_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 1 1 true true true 2 32 true 2 220 100 24 16 1024 4096

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb64_acc4_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 2 64 true 2 220 100 24 16 1024 4096

## GPT-2 + Eager
# 3090

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb8_acc4_1n1g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 1 0 127.0.0.1 1 1 false true true 2 8 false 0 220 100 24 16 1024 4096

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb32_acc4_1n4g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 4 0 127.0.0.1 1 1 false true true 2 32 false 0 220 100 24 16 1024 4096

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb2_gb64_acc4_1n8g
bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 1 false true true 2 64 false 0 220 100 24 16 1024 4096

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
ONEFLOW_MODELS_COMMIT=$(git log --pretty=format:"%H" -n 1)
mv ./test_logs/$HOSTNAME/ ./master

python3 extract_result.py --test_commits $ONEFLOW_COMMIT --test_logs master/$GPU_NAME --models_commit $ONEFLOW_MODELS_COMMIT

/home/ouyangyu/ossutil64 -c /home/ouyangyu/.ossutilconfig cp -f -r master oss://oneflow-test/OneAutoTest/onebench/libai/master
