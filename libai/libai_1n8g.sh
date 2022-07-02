# /bash/bin
set -ex
python3 -m pip install pybind11
python3 -m pip install -e . --user

# 1n8g

# 数据并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 "True" "False" 16 128
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 "True" "True" 128 1024
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 "True" "True" 128 8192

# 模型并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 "True" "False" 32 32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 "True" "True" 128 1024

# 流水并行
bash tools/args_libai_bert_nl48.sh configs/bert_nl48_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 "True" "True" 64 1024
bash tools/args_libai_bert_nl48.sh configs/bert_nl48_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 "True" "True" 96 1536

# 2-D并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 "True" "True" 128 4096
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 "True" "True" 128 512
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 4 "True" "True" 128 2048
