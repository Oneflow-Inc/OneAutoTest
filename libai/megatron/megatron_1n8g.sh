# /bash/bin
set -ex

# 1n8g

# 数据并行
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true false 16 128
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 128 1024
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 128 8192

# 模型并行
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 8 1 true false 32 32
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 8 1 true true 128 1024

# 流水并行
bash examples/megatron_args_pretrain_bert_nl48.sh 1 8 0 127.0.0.1 1 8 true true 64 1024
bash examples/megatron_args_pretrain_bert_nl48.sh 1 8 0 127.0.0.1 1 8 true true 96 1536

# 2-D并行
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true true 128 4096
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true true 128 512
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 4 true true 128 2048
