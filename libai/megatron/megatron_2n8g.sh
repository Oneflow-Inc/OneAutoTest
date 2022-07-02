# /bash/bin
set -ex

# 2n8g

# 数据并行
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 true false 16 256 
sleep 5s
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 1 true true 128 8192 
sleep 5s

# 2-D并行
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 4 1 true true 128 2048
sleep 5s
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 1 true true 128 8192 
sleep 5s
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 8 1 true true 128 2048 
sleep 5s
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 1 4 true true 128 4096

# 3-D并行
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 128 2048
sleep 5s
bash examples/megatron_args_pretrain_bert.sh 2 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 128 4096
