# /bash/bin
set -ex

# 1n1g & 1n4g

# 单卡
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true false 24 24
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 128 1024
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 128 128

# 模型并行
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 4 1 true false 24 24
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 4 1 true true 128 1024

# 流水并行
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 1 4 true true 128 1024
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 1 4 true true 196 1568
bash examples/megatron_args_pretrain_bert_nl48.sh 1 4 0 127.0.0.1 1 4 true true 64 512

# 2-D并行
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 2 1 true true 128 2048
