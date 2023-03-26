!/bin/bash
source /home/oyy/miniconda3/etc/profile.d/conda.sh
conda activate py37
GPU_ID=1
git clone https://github.com/Oneflow-Inc/one-codegeex.git
cd one-codegeex
pip install -e .
pip install torch
pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117


echo "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))" | cat - tests/test_inference.py > temp && mv temp tests/test_inference.py
echo "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))" | cat - tests/test_inference_oneflow.py > temp && mv temp tests/test_inference_oneflow.py
echo "import sys, os" | cat - tests/test_inference.py > temp && mv temp tests/test_inference.py
echo "import sys, os" | cat - tests/test_inference_oneflow.py > temp && mv temp tests/test_inference_oneflow.py
sed -i 's|<path_to_weights>|/data/home/codegeex_13b.pt|g' configs/codegeex_13b.sh
sed -i 's|default=39,|default=40,|g' tests/test_inference_oneflow.py
sed -i '129,130s|state_dict.*|pass|g' tests/test_inference_oneflow.py
sed -i '134s|model.load_state_dict(state_dict)|pass|g' tests/test_inference_oneflow.py
sed -i '204i \    import os\n\    cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"\n\    os.system(cmd)\n' tests/test_inference_oneflow.py
sed -i '201i \    import os\n\    cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"\n\    os.system(cmd)\n' tests/test_inference.py
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


docker pull nvcr.io/nvidia/pytorch:21.11-py3
cd ..
WORK_DIR=$(pwd)
DOCKER_NAME=$(openssl rand -hex 10)
git clone https://github.com/CodeGeeX/codegeex-fastertransformer.git

docker run -p 9100:5008 --cpus 12 --gpus '"device=0"' -it -d -v $WORK_DIR/codegeex-fastertransformer:/workspace/codegeex-fastertransformer --ipc=host --name=$DOCKER_NAME nvcr.io/nvidia/pytorch:21.11-py3

DOCKER_ID=$(docker ps -q --filter "name=$DOCKER_NAME")
docker cp /data/home/ouyangyu/codegeex/codegeex-fastertransformer/codegeex_13b_ft.pt $DOCKER_ID:/workspace/
docker exec -it $DOCKER_ID /bin/bash -c "cd codegeex-fastertransformer && \
                               python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
                               pip3 install transformers sentencepiece && \
                               sh make_all.sh && \
                               python3 api.py --output_len 2048 --ckpt_path /workspace/codegeex_13b_ft.pt --lib_path /workspace/codegeex-fastertransformer/build/lib/libth_codegeex.so &"
docker logs -f $DOCKER_ID &
for length in 128 256 512 1024 2048
do 
    echo "Running for output length: $length"
    for ((i=1; i<=10; i++)); do
        echo "Iteration: $i"
        docker exec -it $DOCKER_ID bash -c "cd /workspace/codegeex-fastertransformer && \
                               python3 post.py --output_len $length 2>&1 | tee -a ${length}_faster_transformer_run_${i}.log"
        nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
        echo "------------------------$length--------------------------"
    done
    sleep 20s
done

python3 extract_log.py