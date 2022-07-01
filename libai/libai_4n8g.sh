# /bash/bin
set -ex
python3 -m pip install pybind11
python3 -m pip install -e . --user

# 4n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 "True" "True" 196 6272
sleep 10s
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 4 4 "True" "True" 256 4096
