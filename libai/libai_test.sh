

https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/commit/01b1d3222c4fd1bb38a2241d9f8ae7cf38e8e532/cu112/oneflow-0.7.0%2Bcu112.git.01b1d32-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# BERT


# bash tools/args_libai_bert.sh model_config pre_gpu node rank master_ip mp pp fp16 activation mbsz gbsz commit

# FP16_activationFalse_mp1_pp1_mb4_gb4_1n1g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True False 4 4 01b1d32

sleep 100s

# FP16_activationTrue_mp1_pp1_mb4_gb32_1n1g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True True 4 32 01b1d32

sleep 100s
# FP16_activationFalse_mp1_pp1_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 192.168.0.10 1 1 True False 4 32 01b1d32

sleep 100s

# FP16_activationFalse_mp2_pp1_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 True False 4 16 01b1d32
sleep 100s

# FP16_activationTrue_mp1_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 2 True True 4 16 01b1d32
sleep 100s

# FP16_activationTrue_mp2_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 4 8 01b1d32
sleep 100s

# FP16_activationTrue_mp2_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 True True 4 32 01b1d32
sleep 100s

# T5

# t5_nl12_nah12_hs76_FP16_activationFalse_mp1_pp1_mb16_gb16_1n1g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 True False 16 16 01b1d32
sleep 100s

# t5_nl12_nah12_hs76_FP16_activationFalse_mp1_pp1_mb16_gb128_1n8g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 True False 16 128 01b1d32
sleep 100s

# t5_nl12_nah12_hs76_FP16_activationFalse_mp2_pp2_mb16_gb128_1n8g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 True True 16 128 01b1d32


sleep 100s


# t5_nl12_nah12_hs76_FP32_activationFalse_mp1_pp1_mb16_gb16_1n1g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 False False 16 16 01b1d32
sleep 100s

# t5_nl12_nah12_hs76_FP32_activationFalse_mp1_pp1_mb16_gb128_1n8g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 False False 16 128 01b1d32
sleep 100s

# t5_nl12_nah12_hs76_FP32_activationFalse_mp2_pp2_mb16_gb128_1n8g

bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 1 1 False True 16 128 01b1d32


sleep 100s



# FP32_activationFalse_mp1_pp1_mb4_gb4_1n1g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 False False 4 4 01b1d32

sleep 100s

# FP32__activationTrue_mp1_pp1_mb4_gb32_1n1g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 False True 4 32 01b1d32

sleep 100s
# FP32_activationFalse_mp1_pp1_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 False False 4 32 01b1d32
sleep 100s

# FP32_activationFalse_mp2_pp1_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 False False 4 16 01b1d32
sleep 100s

# FP32_activationTrue_mp1_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 2 False True 4 16 01b1d32
sleep 100s

# FP32_activationTrue_mp2_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 False True 4 8 01b1d32
sleep 100s

# FP32_activationTrue_mp2_pp2_mb4_gb4_1n8g

bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 False True 4 32 01b1d32
sleep 100s


# gpt


bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 True False 4 4 01b1d32
