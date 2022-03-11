

# BERT acc=4

# bash tools/args_libai_bert.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

 # 1n1g 数据并行

# FP16_activationFalse_mp1_pp1_mb4_gb4

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True False 4 4 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb16
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True True 4 16 277e7a3


# FP16_activationFalse_mp1_pp1_mb8_gb8	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True False 8 8 277e7a3


# FP16_activationTrue_mp1_pp1_mb8_gb64	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True True 8 32 277e7a3

# FP32_activationFalse_mp1_pp1_mb4_gb4
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 False False 4 4 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb32	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 False True 4 32 277e7a3




 # 1n8g 数据并行

# FP16_activationFalse_mp1_pp1_mb4_gb32

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 True False 4 32 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb128
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 True True 4 128 277e7a3


# FP16_activationFalse_mp1_pp1_mb8_gb64	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 True False 8 64 277e7a3


# FP16_activationTrue_mp1_pp1_mb8_gb256	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 True True 8 256 277e7a3

# FP32_activationFalse_mp1_pp1_mb4_gb32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 False False 4 32 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb128
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 False True 4 128 277e7a3



# 1n8g 模型并行

# FP16_activationFalse_mp2_pp1_mb8_gb32	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 True False 8 32 277e7a3


# FP16_activationTrue_mp1_pp2_mb8_gb32	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 2 True True 8 32 277e7a3


# FP16_activationTrue_mp2_pp2_mb8_gb16	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 8 16 277e7a3


# FP16_activationTrue_mp2_pp2_mb8_gb64	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 8 64 277e7a3

#FP16_activationFalse_mp2_pp1_mb4_gb16
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 True False 4 16 277e7a3

#FP16_activationTrue_mp1_pp2_mb4_gb16
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 2 True True 4 16 277e7a3

#FP16_activationTrue_mp2_pp2_mb4_gb8
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 4 8 277e7a3

#FP16_activationTrue_mp2_pp2_mb4_gb32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 4 32 277e7a3


# FP32_activationTrue_mp2_pp2_mb4_gb8
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 False True 4 8 277e7a3

# FP32_activationTrue_mp2_pp2_mb4_gb32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 False True 4 32 277e7a3

# FP32_activationFalse_mp2_pp1_mb4_gb16	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 False False 4 16 277e7a3

# FP32_activationTrue_mp1_pp2_mb4_gb32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 False True 4 32 277e7a3




 # 2n8g 数据并行

# FP16_activationFalse_mp1_pp1_mb4_gb64

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 True False 4 64 277e7a3

# FP16_activationTrue_mp1_pp1_mb4_gb256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 True True 4 256 277e7a3


# FP16_activationFalse_mp1_pp1_mb8_gb64	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 True False 8 128 277e7a3


# FP16_activationTrue_mp1_pp1_mb8_gb512	
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 True True 8 512 277e7a3

# FP32_activationFalse_mp1_pp1_mb4_gb64
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 False False 4 64 277e7a3

# FP32_activationTrue_mp1_pp1_mb4_gb256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 1 1 False True 4 256 277e7a3



# 2n8g 模型并行 acc=4

# FP16_activationTrue_mp2_pp4_mb4_gb16
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 True True 4 8 277e7a3

# FP16_activationTrue_mp2_pp4_mb4_gb64
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 True True 4 32 277e7a3


# FP32_activationTrue_mp2_pp4_mb4_gb8
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 False True 4 8 277e7a3


# FP32_activationTrue_mp2_pp4_mb4_gb32
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 2 8 0 127.0.0.1 2 4 False True 4 32 277e7a3




