# bert

# 仅在火山引擎平台上使用该参数
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  4n8g	3-D并行	bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb192_gb6144_4n8g
bash examples/megatron_args_pretrain_bert.sh 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 2 4 true true 192 6144
sleep 10s

#  4n8g	3-D并行	bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp4_mb256_gb4096_4n8g
bash examples/megatron_args_pretrain_bert.sh 4 8 $MLP_ROLE_INDEX $MLP_WORKER_0_HOST 4 4 true true 256 4096
