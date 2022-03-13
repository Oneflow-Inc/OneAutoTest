

# T5 acc=4

 # 1n1g 数据并行

# FP16_activationFalse_mp1_pp1_mb16_gb16

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 True False 16 16 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb16
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 True True 16 64 277e7a3


# FP32_activationFalse_mp1_pp1_mb4_gb4
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 False False 8 8 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb32  
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 False True 8 32 277e7a3

 # 1n8g 数据并行

# FP16_activationFalse_mp1_pp1_mb4_gb32

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 True False 16 128 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 True True 16 512 277e7a3


# FP32_activationFalse_mp1_pp1_mb4_gb32
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 False False 8 64 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 False True 8 256 277e7a3


# 1n8g 模型并行

# FP16_activationFalse_mp2_pp1_mb8_gb32 
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 1 True False 16 64 277e7a3


# FP16_activationTrue_mp1_pp2_mb8_gb32  
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 2 True True 16 64 277e7a3


# FP16_activationTrue_mp2_pp2_mb8_gb16  
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 True True 16 32 277e7a3


# FP16_activationTrue_mp2_pp2_mb8_gb64  
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 True True 16 64 277e7a3


# FP32_activationTrue_mp2_pp2_mb4_gb8
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 False True 8 16 277e7a3

# FP32_activationTrue_mp2_pp2_mb4_gb32
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 False True 8 32 277e7a3

# FP32_activationFalse_mp2_pp1_mb4_gb16 
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 1 False False 8 32 277e7a3

# FP32_activationTrue_mp1_pp2_mb4_gb32
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 False True 8 32 277e7a3


 # 2n8g 数据并行

# FP16_activationFalse_mp1_pp1_mb4_gb64

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 1 1 True False 16 256 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb256
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 1 1 True True 16 1024 277e7a3


# FP32_activationFalse_mp1_pp1_mb4_gb64
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 1 1 False False 8 128 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb256
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 1 1 False True 8 512 277e7a3



# 2n8g 模型并行 acc=4

# FP16_activationTrue_mp2_pp4_mb4_gb16
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 2 4 True True 16 32 277e7a3

# FP16_activationTrue_mp2_pp4_mb4_gb64
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 2 4 True True 16 128 277e7a3


# FP32_activationTrue_mp2_pp4_mb4_gb8
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 2 4 False True 8 16 277e7a3


# FP32_activationTrue_mp2_pp4_mb4_gb32
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 2 8 0 127.0.0.1 2 4 False True 8 64 277e7a3
