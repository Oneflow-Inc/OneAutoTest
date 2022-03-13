

# GPT acc=4

 # 1n1g 数据并行

# FP16_activationfalse_mp1_pp1_mb16_gb16

bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true false 16 16 277e7a3

# FP16_activationtrue_mp1_pp1_mb4_gb16
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 16 64 277e7a3


# FP32_activationfalse_mp1_pp1_mb4_gb4
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 false false 8 8 277e7a3

# FP32_activationtrue_mp1_pp1_mb4_gb32  
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 false true 8 32 277e7a3

 # 1n8g 数据并行

# FP16_activationfalse_mp1_pp1_mb4_gb32

bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true false 16 128 277e7a3

# FP16_activationtrue_mp1_pp1_mb4_gb128
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 16 512 277e7a3


# FP32_activationfalse_mp1_pp1_mb4_gb32
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 false false 8 64 277e7a3

# FP32_activationtrue_mp1_pp1_mb4_gb128
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 false true 8 256 277e7a3


# 1n8g 模型并行

# FP16_activationfalse_mp2_pp1_mb8_gb32 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true false 16 64 277e7a3


# FP16_activationtrue_mp1_pp2_mb8_gb32  
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 2 true true 16 64 277e7a3


# FP16_activationtrue_mp2_pp2_mb8_gb16  
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 true true 16 32 277e7a3


# FP16_activationtrue_mp2_pp2_mb8_gb64  
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 true true 16 64 277e7a3


# FP32_activationtrue_mp2_pp2_mb4_gb8
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 false true 8 16 277e7a3

# FP32_activationtrue_mp2_pp2_mb4_gb32
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 false true 8 32 277e7a3

# FP32_activationfalse_mp2_pp1_mb4_gb16 
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 false false 8 32 277e7a3

# FP32_activationtrue_mp1_pp2_mb4_gb32
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 false true 8 32 277e7a3


 # 2n8g 数据并行

# FP16_activationfalse_mp1_pp1_mb4_gb64

bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 true false 16 256 277e7a3

# FP16_activationtrue_mp1_pp1_mb4_gb256
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 true true 16 1024 277e7a3


# FP32_activationfalse_mp1_pp1_mb4_gb64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 false false 8 128 277e7a3

# FP32_activationtrue_mp1_pp1_mb4_gb256
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 false true 8 512 277e7a3



# 2n8g 模型并行 acc=4

# FP16_activationtrue_mp2_pp4_mb4_gb16
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 true true 16 32 277e7a3

# FP16_activationtrue_mp2_pp4_mb4_gb64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 true true 16 128 277e7a3


# FP32_activationtrue_mp2_pp4_mb4_gb8
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 false true 8 16 277e7a3


# FP32_activationtrue_mp2_pp4_mb4_gb32
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 false true 8 64 277e7a3
