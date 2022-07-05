# GPT2

# 仅在火山引擎平台上使用该参数
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  2n8g	数据并行	gpt_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb16_gb256_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 1 1 true false 4 64
sleep 10s

#  2n8g	数据并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb8192_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 1 1 true true 32 2048
sleep 10s

#  2n8g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb128_gb2048_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 4 1 true true 32 512
sleep 10s

#  2n8g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb8192_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 2 1 true true 32 2048
sleep 10s

#  2n8g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb128_gb2048_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 8 1 true true 32 512
sleep 10s

#  2n8g	数据+流水并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb128_gb4096_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 1 4 true true 32 1024
sleep 10s

#  2n8g	3-D并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb128_gb2048_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 2 4 true true 32 512
sleep 10s

#  2n8g	3-D并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb128_gb4096_2n8g
bash examples/megatron_args_pretrain_gpt2.sh 2 8 $NODE_RANK $MASTER_ADDR 2 4 true true 32 1024
