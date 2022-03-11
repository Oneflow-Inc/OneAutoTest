

# BERT acc=4

# bash tools/args_libai_bert.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

 # 1n1g 数据并行

# FP16_activationfalse_mp1_pp1_mb4_gb4

bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true false 4 4

# FP16_activationtrue_mp1_pp1_mb4_gb16
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 4 16


# FP16_activationfalse_mp1_pp1_mb8_gb8  
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true false 8 8


# FP16_activationtrue_mp1_pp1_mb8_gb64  
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 true true 8 32

# FP32_activationfalse_mp1_pp1_mb4_gb4
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 false false 4 4

# FP32_activationtrue_mp1_pp1_mb4_gb32  
bash examples/megatron_args_pretrain_bert.sh 1 1 0 127.0.0.1 1 1 false true 4 32




 # 1n8g 数据并行

# FP16_activationfalse_mp1_pp1_mb4_gb32

bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true false 4 32

# FP16_activationtrue_mp1_pp1_mb4_gb128
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 4 128


# FP16_activationfalse_mp1_pp1_mb8_gb64 
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true false 8 64


# FP16_activationtrue_mp1_pp1_mb8_gb256 
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 true true 8 256

# FP32_activationfalse_mp1_pp1_mb4_gb32
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 false false 4 32

# FP32_activationtrue_mp1_pp1_mb4_gb128
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 1 false true 4 128



# 1n8g 模型并行

# FP16_activationfalse_mp2_pp1_mb8_gb32 
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true false 8 32


# FP16_activationtrue_mp1_pp2_mb8_gb32  
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 2 true true 8 32


# FP16_activationtrue_mp2_pp2_mb8_gb16  
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 true true 8 16


# FP16_activationtrue_mp2_pp2_mb8_gb64  
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 true true 8 64

#FP16_activationfalse_mp2_pp1_mb4_gb16
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 true false 4 16

#FP16_activationtrue_mp1_pp2_mb4_gb16
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 1 2 true true 4 16

#FP16_activationtrue_mp2_pp2_mb4_gb8
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 true true 4 8

#FP16_activationtrue_mp2_pp2_mb4_gb32
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 true true 4 32


# FP32_activationtrue_mp2_pp2_mb4_gb8
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 false true 4 8

# FP32_activationtrue_mp2_pp2_mb4_gb32
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 false true 4 32

# FP32_activationfalse_mp2_pp1_mb4_gb16 
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 1 false false 4 16

# FP32_activationtrue_mp1_pp2_mb4_gb32
bash examples/megatron_args_pretrain_bert.sh 1 8 0 127.0.0.1 2 2 false true 4 32




 # 2n8g 数据并行

# FP16_activationfalse_mp1_pp1_mb4_gb64

bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 true false 4 64

# FP16_activationtrue_mp1_pp1_mb4_gb256
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 true true 4 256


# FP16_activationfalse_mp1_pp1_mb8_gb64 
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 true false 8 128


# FP16_activationtrue_mp1_pp1_mb8_gb512 
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 true true 8 512

# FP32_activationfalse_mp1_pp1_mb4_gb64
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 false false 4 64

# FP32_activationtrue_mp1_pp1_mb4_gb256
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 1 1 false true 4 256



# 2n8g 模型并行 acc=4

# FP16_activationtrue_mp2_pp4_mb4_gb16
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 2 4 true true 4 8

# FP16_activationtrue_mp2_pp4_mb4_gb64
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 2 4 true true 4 32


# FP32_activationtrue_mp2_pp4_mb4_gb8
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 2 4 false true 4 8


# FP32_activationtrue_mp2_pp4_mb4_gb32
bash examples/megatron_args_pretrain_bert.sh 2 8 0 127.0.0.1 2 4 false true 4 32
