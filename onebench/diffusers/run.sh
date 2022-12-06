set -ex
#env install
#    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -P ./
#    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh
#    conda create -n py38 python=3.8
#    conda activate py38

# 模型下载
# user_header="Authorization: Bearer xxxxxx"
# wget --header="${user_header}" https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O models/sd-v1-4.ckpt


GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

if [ ! -d "./test_logs/$GPU_NAME" ]; then
  mkdir -p ./test_logs/$GPU_NAME 
fi
cd test_logs
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112



# 两个问题
#  1. 要安装pytorch
#  2. 先已经安装了transformers，因此需要先安装transformers
python3 -m pip uninstall -y diffusers transformers
if [ ! -d "./transformers" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/transformers.git
  cd transformers
  python3 -m pip install -e .
  cd ..
fi
if [ ! -d "./diffusers" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/diffusers.git
  cd diffusers
  python3 -m pip install -e .[oneflow]
  cd ..
fi

export HUGGING_FACE_HUB_TOKEN=xxxx
export HF_HOME=/hf/home/


# mkdir $GPU_NAME

python3 ../oneflow_args.py \
--model_id "CompVis/stable-diffusion-v1-4" \
--num_images_per_prompt 20 \
--num_inference_steps 50 \
--img_height 512 \
--img_width 512 \
--scheduler "ddpm" \
--saving_path "oneflow-sd-output" 2>&1 | tee $GPU_NAME/oneflow-args.log


### pytorch 
python3 -m pip uninstall -y diffusers transformers

rm -rf ./diffusers
git clone --depth 1 https://github.com/huggingface/diffusers.git
cd diffusers
python3 -m pip install transformers
python3 -m pip install --upgrade diffusers[torch]
cd ..

# mkdir $GPU_NAME

python3 ../pytorch_args.py \
--model_id "CompVis/stable-diffusion-v1-4" \
--num_images_per_prompt 20 \
--num_inference_steps 50 \
--img_height 512 \
--img_width 512 \
--scheduler "ddpm" \
--saving_path "pytorch-sd-output" 2>&1 | tee $GPU_NAME/pytorch-args.log


