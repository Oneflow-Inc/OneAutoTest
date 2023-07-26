set -ex
source /data/home/sunjinfeng/miniconda3/etc/profile.d/conda.sh

conda activate run_val

cd libai
# 安装 libai
python3 -m pip uninstall -y libai
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --user
python3 -m pip install -e . --user

bash tools/args_train.sh configs/gpt2_pretrain.py 1 8 0 127.0.0.1 1 1 true true true 8 512 true 2 220 100 24 16 1024 4096
sleep 2m

DATA_PATH=${DATA_PATH:-"/ssd/dataset/ImageNet/extract"}

sed -i "s+/path/to/imagenet+${DATA_PATH}+g" ./configs/swin_imagenet.py

bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 2 true true true 128 2048 true 2 220 100

python3 -m pip uninstall -y oneflow

python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/commit/fb423fa80073e054cb8f5c33606457f5da62c0a2/cu117/

bash tools/args_train.sh configs/bert_large_pretrain.py 1 8 0 127.0.0.1 2 2 true true true 32 512 true 2 220 100 24 16 1024 4096
sleep 2m

#  3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n8g
bash tools/args_train.sh configs/swin_imagenet.py 1 8 0 127.0.0.1 2 2 true true true 128 2048 true 2 220 100
sleep 2m
