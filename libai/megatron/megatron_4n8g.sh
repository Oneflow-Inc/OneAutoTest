# /bash/bin
set -ex

# 4n8g

bash examples/megatron_args_pretrain_bert.sh 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 196 6272
sleep 10s
bash examples/megatron_args_pretrain_bert.sh 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 4 4 true true 256 4096
