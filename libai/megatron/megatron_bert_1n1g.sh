## BERT
#  1n1g		bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb24_gb24_1n1g

bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true false 24 24

#  1n1g		bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb1024_1n1g
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 128 1024

#  1n1g		bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb128_1n1g
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 128 128

#  1n4g	模型并行	bert_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb24_gb24_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 4 1 true false 24 24

#  1n4g	模型并行	bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb128_gb1024_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 4 1 true true 128 1024

#  1n4g	流水并行	bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb128_gb1024_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 1 4 true true 128 1024

#  1n4g	流水并行	bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb192_gb1536_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 1 4 true true 192 1536

#  1n4g	流水并行	bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb64_gb512_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 1 4 true true 64 512 48

#  1n4g	数据+模型并行	bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb2048_1n4g
bash examples/megatron_args_pretrain_bert.sh 1 4 0 127.0.0.1 2 1 true true 128 2048
