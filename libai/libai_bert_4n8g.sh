## BERT

# volcengine.com
NODE_RANK=$MLP_ROLE_INDEX
MASTER_ADDR=$MLP_WORKER_0_HOST

#  4n8g 数据并行 bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb16_gb512_4n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 1 true false 16 512

#  4n8g 数据并行 bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb4096_4n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 1 true true 128 4096

#  4n8g 数据+流水并行 bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb128_gb8192_4n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 1 4 true true 128 8192

#  4n8g 3-D并行 bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp4_mb192_gb6144_4n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 2 4 true true 192 6144

#  4n8g 3-D并行 bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp4_mb256_gb4096_4n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 4 8 $NODE_RANK $MASTER_ADDR 4 4 true true 256 4096
