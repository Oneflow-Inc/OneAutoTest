## GPT-2

# 仅在火山引擎上使用该参数
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  4n8g 3-D并行 gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb48_gb1536_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 2 4 true true 48 1536
sleep 10s

#  4n8g 3-D并行 gpt_nl24_nah16_hs1024_fp16_actrue_mp4_pp4_mb256_gb4096_4n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 4 4 true true 64 1024
