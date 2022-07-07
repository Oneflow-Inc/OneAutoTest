## GPT-2

#  1n1g		gpt_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb6_gb6_1n1g
bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 1 1 true false 6 6

#  1n1g		gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n1g
bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 1 1 true true 32 256

#  1n1g		gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb32_1n1g
bash examples/megatron_args_pretrain_gpt2.sh 1 1 0 127.0.0.1 1 1 true true 32 32

#  1n4g	模型并行	gpt_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb6_gb6_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 4 1 true false 6 6

#  1n4g	模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb256_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 4 1 true true 32 256

#  1n4g	流水并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb256_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 1 4 true true 32 256

#  1n4g	流水并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb48_gb392_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 1 4 true true 48 384

#  1n4g	流水并行	gpt_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb16_gb128_1n4g
bash examples/megatron_args_pretrain_gpt2_nl48.sh 1 4 0 127.0.0.1 1 4 true true 16 128 48

#  1n4g 数据并行    gpt_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb16_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 1 1 true false 4 16

#  1n4g 数据并行    gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 1 1 true true 32 128

#  1n4g	数据+模型并行	gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb512_1n4g
bash examples/megatron_args_pretrain_gpt2.sh 1 4 0 127.0.0.1 2 1 true true 32 512
