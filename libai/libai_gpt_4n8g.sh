## GPT-2

# volcengine.com
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  4n8g 数据并行 gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb128_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 1 true false 4 128

#  4n8g 数据并行 gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb1024_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 1 true true 32 1024

#  4n8g 数据+流水并行 gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb2048_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 4 true true 32 2048

#  4n8g 3-D并行 gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb48_gb1536_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 2 4 true true 48 1536

#  4n8g 3-D并行 gpt2_nl24_nah16_hs1024_fp16_actrue_mp4_pp4_mb64_gb1024_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 4 4 true true 64 1024
