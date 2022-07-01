# /bash/bin
set -ex
python3 -m pip install pybind11
python3 -m pip install -e . --user

# 2n8g

# 数据并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 "True" "False" 16 256 
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 "True" "True" 128 8192 
sleep 5s

# 2-D并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 4 1 "True" "True" 128 2048
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 1 "True" "True" 128 8192 
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 8 1 "True" "True" 128 2048 
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 4 "True" "True" 128 4096

# 3-D并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 "True" "True" 128 2048
sleep 5s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 "True" "True" 128 4096
