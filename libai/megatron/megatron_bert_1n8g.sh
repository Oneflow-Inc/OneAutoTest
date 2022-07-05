# bert

#  1n8g	模型并行	bert_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb32_gb32_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 8 1 true false 32 32

#  1n8g	模型并行	bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb128_gb1024_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 8 1 true true 128 1024

#  1n8g	流水并行	bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb64_gb1024_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 8 true true 64 1024 48

#  1n8g	流水并行	bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb96_gb1536_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 8 true true 96 1536 48

#  1n8g	数据并行	bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb16_gb128_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true false 16 128

#  1n8g	数据并行	bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb1024_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 128 1024

#  1n8g	数据并行	bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb8192_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 128 8192

#  1n8g	数据+模型并行	bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb4096_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true true 128 4096

#  1n8g	数据+模型并行	bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb512_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true true 128 512

#  1n8g	数据+流水并行	bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb128_gb2048_1n8g
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 4 true true 128 2048
