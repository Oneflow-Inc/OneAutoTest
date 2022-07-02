# /bash/bin
set -ex
python3 -m pip install pybind11
python3 -m pip install -e . --user

# 1n1g & 1n4g

# 单卡
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 "True" "False" 24 24
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 "True" "True" 128 1024
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 "True" "True" 128 128

# 模型并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 "True" "False" 24 24
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 "True" "True" 128 1024

# 流水并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 "True" "True" 128 1024
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 "True" "True" 196 1568
bash tools/args_libai_bert_nl48.sh configs/bert_nl48_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 "True" "True" 64 512

# 2-D并行
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 "True" "True" 128 2048
