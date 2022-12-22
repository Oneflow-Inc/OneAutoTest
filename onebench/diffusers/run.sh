set -ex
#env install
#    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -P ./
#    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh
#    conda create -n py38 python=3.8
#    conda activate py38

# 模型下载
# user_header="Authorization: Bearer xxxxxx"
# wget --header="${user_header}" https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O models/sd-v1-4.ckpt
STABLE_VERSION=${1:-"sdv1.4"} # sdv1.4 sdv2
SCHEDULER=${2:-""}

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"

if [ ! -d "./test_logs" ]; then
  mkdir -p ./test_logs
fi
cd test_logs
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python3 -m pip uninstall -y oneflow
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112

export HUGGING_FACE_HUB_TOKEN=xxxx
export HF_HOME=/hf/home

# 两个问题
#  1. 要安装pytorch
#  2. 先已经安装了transformers，因此需要先安装transformers
python3 -m pip uninstall -y diffusers transformers
if [ ! -d "./transformers" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/transformers.git
fi

cd transformers
python3 -m pip install -e .
cd ..

if [ ! -d "./diffusers" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/diffusers.git
fi
cd diffusers
python3 -m pip install -e .[oneflow]
cd ..


NUM_IMAGES_PER_PROMPT=20
NUM_INFERENCE_STEPS=50
IMG_HEIGHT=512
IMG_WIDTH=512
MODEL_ID_NAME="CompVis/stable-diffusion-v1-4"

if [ "$SCHEDULER" == "dmp" ]; then
    NUM_INFERENCE_STEPS=20
fi
if [ "$STABLE_VERSION" == "sdv2" ]; then
    MODEL_ID_NAME="stabilityai/stable-diffusion-2"
    IMG_HEIGHT=768
    IMG_WIDTH=768
fi

if [ ! -d "./stable_logs" ]; then
  mkdir stable_logs
fi


wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/diffusers/args_stable_diffusion.py

CMD=""
CMD+="python3 args_stable_diffusion.py "
CMD+="--model_id $MODEL_ID_NAME "
CMD+="--num_images_per_prompt $NUM_IMAGES_PER_PROMPT "
CMD+="--num_inference_steps $NUM_INFERENCE_STEPS "
CMD+="--img_height $IMG_HEIGHT "
CMD+="--img_width $IMG_WIDTH "

if [ -n "$SCHEDULER" ]; then
    CMD+="--scheduler $SCHEDULER "
fi


# run oneflow
DL_FRAME="oneflow"
LOG_FILENAME=stable_logs/${GPU_NAME}_${DL_FRAME}_${NUM_INFERENCE_STEPS}_HEIGHT${IMG_WIDTH}X${IMG_HEIGHT}_${STABLE_VERSION}_${NUM_IMAGES_PER_PROMPT}
DL_FRAME="${CMD} --dl_frame $DL_FRAME --saving_path $LOG_FILENAME "
echo "Rum ${DL_FRAME} cmd ${DL_FRAME}"

$DL_FRAME 2>&1 | tee ${LOG_FILENAME}.log

### pytorch 
DL_FRAME="pytorch"

python3 -m pip uninstall -y diffusers transformers
python3 -m pip install transformers

if [ ! -d "./${DL_FRAME}/diffusers" ]; then
  mkdir -p ./${DL_FRAME}/diffusers
  git clone --depth 1 https://github.com/huggingface/diffusers.git ./${DL_FRAME}/diffusers
fi
cd ./${DL_FRAME}/diffusers
python3 -m pip install --upgrade diffusers[torch]
cd -

LOG_FILENAME=stable_logs/${GPU_NAME}_${DL_FRAME}_${NUM_INFERENCE_STEPS}_HEIGHT${IMG_WIDTH}X${IMG_HEIGHT}_${STABLE_VERSION}_${NUM_IMAGES_PER_PROMPT}
DL_FRAME="${CMD} --dl_frame $DL_FRAME --saving_path $LOG_FILENAME "
echo "Rum ${DL_FRAME} cmd ${DL_FRAME}"


$DL_FRAME 2>&1 | tee ${LOG_FILENAME}.log

python3 -m pip uninstall -y diffusers transformers
