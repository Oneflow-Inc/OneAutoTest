set -ex
#env install
#    wget -nc https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -P ./
#    bash Miniconda3-py39_4.12.0-Linux-x86_64.sh
#    conda create -n py38 python=3.8
#    conda activate py38

# Model download
# user_header="Authorization: Bearer xxxxxx"
# wget --header="${user_header}" https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O models/sd-v1-4.ckpt

STABLE_VERSION=${1:-"sdv1_5"} # sdv1_5 sdv2_0 sdv2_1 taiyi
INSTALL_ONEFLOW=${2:-"master"}
CUDA_VERSION=${3:-"cu117"}

SRC_DIR=$(realpath $(dirname $0))
mkdir -p ${SRC_DIR}/hf/home
export HUGGING_FACE_HUB_TOKEN=hf_
export HF_HOME=${SRC_DIR}/hf/home

# install oneflow 
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

python3 -m pip install sentencepiece click 

if [ "$INSTALL_ONEFLOW" != "false" ]; then
  python3 -m pip uninstall -y oneflow
  if [ "$INSTALL_ONEFLOW" != "master" ]; then
    python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/${CUDA_VERSION}/${INSTALL_ONEFLOW}/index.html
  else
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/${INSTALL_ONEFLOW}/${CUDA_VERSION}
  fi
fi

# install torch
python3 -m pip install "torch<2" "transformers>=4.26" "diffusers[torch]==0.14.0"

declare -A STABLE_DIFFUSION_SCRIPTS=(
  [sdv2_1]=stable_diffusion_2_1.py
  [sdv2_0]=stable_diffusion_2.py
  [sdv1_5]=stable_diffusion_v1_5.py
  [taiyi]=taiyi_stable_diffusion_chinese.py
  [alt_m9]=alt_diffusion_m9.py
)
BENCHMARK_SCRIPT=${STABLE_DIFFUSION_SCRIPTS["$STABLE_VERSION"]}
echo "using $BENCHMARK_SCRIPT benchmark script"

if [ "$BENCHMARK_SCRIPT" == "" ]; then
  exit
fi

python3 -m pip uninstall -y diffusers

if [ ! -d "./diffusion-benchmark" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/diffusion-benchmark.git
fi
cd diffusion-benchmark
DIFFUSION_BENCHMARK_COMMIT=$(git log --pretty=format:"%H" -n 1)
echo "diffusion-benchmark(git_commit)=$DIFFUSION_BENCHMARK_COMMIT"


if [ ! -d "./diffusers" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/diffusers.git
fi
cd diffusers
ONEFLOW_DIFFUSERS_COMMIT=$(git log --pretty=format:"%H" -n 1)
echo "oneflow-diffusers(git_commit)=$ONEFLOW_DIFFUSERS_COMMIT"
python3 -m pip install -e .[oneflow]
cd ..

if [ "$STABLE_VERSION" == "sdv2_0" ]; then
    sed -i '/import oneflow as torch/a\torch.mock_torch.enable()' ./scripts/$BENCHMARK_SCRIPT
    sed -i '/from diffusers import (/,/)/c\
from diffusers import EulerDiscreteScheduler\' ./scripts/$BENCHMARK_SCRIPT
    sed -i '/from diffusers/a\from onediff import OneFlowStableDiffusionPipeline as StableDiffusionPipeline' ./scripts/$BENCHMARK_SCRIPT
fi

if [ "$STABLE_VERSION" == "sdv1_5" ]; then
    sed -i '/import oneflow as torch/a\torch.mock_torch.enable()' ./scripts/$BENCHMARK_SCRIPT
    sed -i 's/from diffusers/from onediff/g' ./scripts/$BENCHMARK_SCRIPT
fi

if [ "$STABLE_VERSION" == "taiyi" ]; then
    sed -i 's/from diffusers/from onediff/g' ./scripts/$BENCHMARK_SCRIPT
fi

sed -i '/for r in range(repeat):/a\
        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
        os.system(cmd)' ./scripts/$BENCHMARK_SCRIPT

if [ "$STABLE_VERSION" == "sdv1_5" ] || [ "$STABLE_VERSION" == "taiyi" ]; then
    sed -i '/@click.option("--output", default="output")/a\@click.option("--height", default=512)\
@click.option("--width", default=512)' ./scripts/$BENCHMARK_SCRIPT
    sed -i 's/images = pipe(prompt).images/images = pipe(prompt, height=height, width=width).images/g' ./scripts/$BENCHMARK_SCRIPT
    sed -i 's/def benchmark(token, prompt, repeat, output):/def benchmark(token, prompt, repeat, output, height, width):/g' ./scripts/$BENCHMARK_SCRIPT
fi


GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"


NUM_REAT=32
IMG_HEIGHT=512
IMG_WIDTH=512
if [ "$STABLE_VERSION" == "sdv2_0" ] || [ "$STABLE_VERSION" == "sdv2_1" ]; then
  IMG_HEIGHT=768
  IMG_WIDTH=768
fi

CMD=""
CMD+="python3 scripts/$BENCHMARK_SCRIPT --repeat 10 --height $IMG_HEIGHT --width $IMG_WIDTH"

# run oneflow
DL_FRAME="oneflow"
LOG_FOLDER=stable_logs/$GPU_NAME/$DL_FRAME
ONEFLOW_VERSION=$(python3 -c 'import oneflow; print(oneflow.__version__)')
ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')


mkdir -p $LOG_FOLDER/$ONEFLOW_COMMIT

LOG_FILENAME=$LOG_FOLDER/$ONEFLOW_COMMIT/${GPU_NAME}_${DL_FRAME}_${BENCHMARK_SCRIPT}_${ONEFLOW_COMMIT}
DL_FRAME="${CMD} --output $LOG_FILENAME "

# remove
python3 -m pip uninstall -y accelerate

$DL_FRAME 2>&1 | tee ${LOG_FILENAME}.log

# echo 
echo "oneflow-version(git_commit)=$ONEFLOW_VERSION" >> ${LOG_FILENAME}.log
echo "oneflow-commit(git_commit)=$ONEFLOW_COMMIT" >> ${LOG_FILENAME}.log
echo "diffusion-benchmark(git_commit)=$DIFFUSION_BENCHMARK_COMMIT" >> ${LOG_FILENAME}.log
#echo "oneflow-transformers(git_commit)=$ONEFLOW_TRANSFORMERS_COMMIT" >> ${LOG_FILENAME}.log
echo "oneflow-diffusers(git_commit)=$ONEFLOW_DIFFUSERS_COMMIT" >> ${LOG_FILENAME}.log

### pytorch 
DL_FRAME="pytorch"
LOG_FOLDER=stable_logs/$GPU_NAME/$DL_FRAME

git checkout ./scripts/$BENCHMARK_SCRIPT
sed -i 's/oneflow as //g' ./scripts/$BENCHMARK_SCRIPT
sed -i 's/OneFlow//g' ./scripts/$BENCHMARK_SCRIPT
sed -i '/for r in range(repeat):/a\
        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
        os.system(cmd)' ./scripts/$BENCHMARK_SCRIPT

TORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')
python3 -m pip uninstall -y diffusers

if [ ! -d "./${DL_FRAME}/diffusers" ]; then
  mkdir -p ./${DL_FRAME}/diffusers
  git clone --depth 1 https://github.com/huggingface/diffusers.git ./${DL_FRAME}/diffusers
fi
cd ./${DL_FRAME}/diffusers
HUGGINGFACE_DIFFUSERS_COMMIT=$(git log --pretty=format:"%H" -n 1)
echo "huggingface-diffusers(git_commit)=$HUGGINGFACE_DIFFUSERS_COMMIT"
python3 -m pip install --upgrade diffusers[torch]
cd -

mkdir -p $LOG_FOLDER/$TORCH_VERSION
LOG_FILENAME=$LOG_FOLDER/$TORCH_VERSION/${GPU_NAME}_${DL_FRAME}_${BENCHMARK_SCRIPT}
DL_FRAME="${CMD} --output $LOG_FILENAME "
echo "Rum ${DL_FRAME} cmd ${DL_FRAME}"


$DL_FRAME 2>&1 | tee ${LOG_FILENAME}.log
# echo 
echo "pytorch-version(git_commit)=$TORCH_VERSION" >> ${LOG_FILENAME}.log
echo "diffusion-benchmark(git_commit)=$DIFFUSION_BENCHMARK_COMMIT" >> ${LOG_FILENAME}.log
echo "huggingface-diffusers(git_commit)=$HUGGINGFACE_DIFFUSERS_COMMIT" >> ${LOG_FILENAME}.log


git checkout ./scripts/$BENCHMARK_SCRIPT

python3 -m pip uninstall -y diffusers transformers
