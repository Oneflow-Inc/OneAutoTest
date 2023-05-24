#!/bin/bash
conda init bash
source /opt/conda/etc/profile.d/conda.sh
if conda env list | grep -q '^py37\s'; then
    echo "Environment 'py37' exists. Activating it now."
    conda activate py37
else
    echo "Environment 'py37' does not exist. Creating it from 'environment.yml'."
    conda env create -f environment.yml
    conda activate py37
fi
GPU_ID=0
git clone https://github.com/Oneflow-Inc/one-codegeex.git
cd one-codegeex
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
pip install torch
pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
pip install cpm_kernels
pip install deepspeed
pip install transformers
pip install xgboost

echo "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))" | cat - tests/test_inference.py > temp && mv temp tests/test_inference.py
echo "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))" | cat - tests/test_inference_oneflow.py > temp && mv temp tests/test_inference_oneflow.py
echo "import sys, os" | cat - tests/test_inference.py > temp && mv temp tests/test_inference.py
echo "import sys, os" | cat - tests/test_inference_oneflow.py > temp && mv temp tests/test_inference_oneflow.py
cat << 'EOF' > configs/codegeex_13b.sh
# CodeGeeX-13B configuration

CHECKPOINT_PATH="/workspace/codegeex_13b.pt"

MODEL_ARGS="--num-layers 39 \
            --hidden-size 5120 \
            --num-attention-heads 40 \
            --max-position-embeddings 2048 \
            --attention-softmax-in-fp32 \
            --load "$CHECKPOINT_PATH" \
            --layernorm-epsilon 1e-5 \
            --fp16 \
            --ws-encoding-start-id 10 \
            --ws-encoding-length 10 \
            --make-vocab-size-divisible-by 52224 \
            --seq-length 2048"
EOF
sed -i 's|default=39,|default=40,|g' tests/test_inference_oneflow.py
sed -i '129,130s|state_dict.*|pass|g' tests/test_inference_oneflow.py
sed -i '134s|model.load_state_dict(state_dict)|pass|g' tests/test_inference_oneflow.py
sed -i '/print(times)/i \    import os\n    cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"\n    os.system(cmd)' tests/test_inference_oneflow.py
sed -i '/print(times)/i \    import os\n    cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"\n    os.system(cmd)' tests/test_inference.py
sed -i '326s|break|pass|g' codegeex/oneflow/inference.py
sed -i 's|--out-seq-length 1024|--out-seq-length $OUTPUT_LEN|g' scripts/test_inference_oneflow.sh
sed -i '7i OUTPUT_LEN=$3' scripts/test_inference_oneflow.sh
sed -i 's|--out-seq-length 1024|--out-seq-length $OUTPUT_LEN|g' scripts/test_inference.sh
sed -i '7i OUTPUT_LEN=$3' scripts/test_inference.sh

for length in 128 256 512 1024 2048
do
    script_name="test_inference.sh"

    for i in {1..10}
    do
        bash ./scripts/$script_name $GPU_ID ./tests/test_prompt.txt $length 2>&1 | tee ${length}_pytorch_run_${i}.log       
    done
    sleep 60
    script_name="test_inference_oneflow.sh"

    for i in {1..10}
    do
        bash ./scripts/$script_name $GPU_ID ./tests/test_prompt.txt $length 2>&1 | tee ${length}_oneflow_run_${i}.log
    done
    sleep 60

done

cd ..
WORK_DIR=$(pwd)
git clone https://github.com/CodeGeeX/codegeex-fastertransformer.git

cd codegeex-fastertransformer && \
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
pip3 install transformers sentencepiece && \
sh make_all.sh && \
python3 api.py --output_len 2048 --ckpt_path /workspace/codegeex_13b_ft.pt --lib_path /workspace/codegeex-fastertransformer/build/lib/libth_codegeex.so &
FLASK_PID=$!
for length in 128 256 512 1024 2048
do 
    echo "Running for output length: $length"
    for ((i=1; i<=10; i++)); do
        echo "Iteration: $i"
        cd codegeex-fastertransformer && \
        python3 post.py --output_len $length 2>&1 | tee -a ${length}_faster_transformer_run_${i}.log
        nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
        echo "------------------------$length--------------------------"
    done
    sleep 20s
done
kill $FLASK_PID