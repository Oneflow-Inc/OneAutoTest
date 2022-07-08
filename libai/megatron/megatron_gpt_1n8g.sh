# GPT2

#  1n8g	模型并行	gpt_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb8_gb8_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 8 1 true false 8 8

#  1n8g	模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 8 1 true true 32 256

#  1n8g	流水并行	gpt_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n8g
bash examples/megatron_args_pretrain_gpt2_nl48.sh 1 8 0 127.0.0.1 1 8 true true 16 256 48

#  1n8g	流水并行	gpt_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash examples/megatron_args_pretrain_gpt2_nl48.sh 1 8 0 127.0.0.1 1 8 true true 24 384 48

#  1n8g	数据并行	gpt_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 1 1 true false 4 32

#  1n8g	数据并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 1 1 true true 32 256

#  1n8g	数据并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 1 1 true true 32 2048

#  1n8g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 2 1 true true 32 1024

#  1n8g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 2 1 true true 32 128

#  1n8g	数据+流水并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash examples/megatron_args_pretrain_gpt2.sh 1 8 0 127.0.0.1 1 4 true true 32 512

