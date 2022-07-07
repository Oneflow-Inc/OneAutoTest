## GPT-2

# volcengine.com
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  2n8g 数据并行        gpt_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb64_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 1 1 true false 4 64

#  2n8g 数据并行        gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 1 1 true true 32 2048

#  2n8g 数据+模型并行   gpt_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb512_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 4 1 true true 32 512

#  2n8g 数据+模型并行   gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb2048_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 2 1 true true 32 2048

#  2n8g 数据+模型并行   gpt_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb512_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 8 1 true true 32 512

#  2n8g 数据+流水并行   gpt_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb1024_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 1 4 true true 32 1024

#  2n8g 3-D并行 gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb32_gb512_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 2 4 true true 32 512

#  2n8g 3-D并行 gpt_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb32_gb1024_2n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 2 8 $NODE_RANK $MASTER_ADDR 2 4 true true 32 1024
