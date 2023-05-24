COMMIT=${1:-"55b822e"}

## BERT
#  1n1g         bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb6_gb6_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true false 6 6 24 ${COMMIT}

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 32 256 24 ${COMMIT}

#  1n1g         bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb32_1n1g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 32 32 24 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb6_gb6_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true false 6 6 24 ${COMMIT}

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb32_gb256_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 32 256 24 ${COMMIT}

#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb256_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 32 256 24 ${COMMIT}

#  1n4g 流水并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb48_gb384_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 48 384 24 ${COMMIT}

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb16_gb128_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 16 128 48 ${COMMIT}

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb16_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true false 4 16 24 ${COMMIT}

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 32 128 24 ${COMMIT}

#  1n4g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb512_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 32 512 24 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb32_gb32_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true false 32 32 24 ${COMMIT}

#  1n8g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true true 32 256 24 ${COMMIT}

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 16 256 48 ${COMMIT}

#  1n8g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb24_gb384_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 24 384 48 ${COMMIT}

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true false 4 32 24 ${COMMIT}

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb256_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 32 256 24 ${COMMIT}

#  1n8g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb2048_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 32 2048 24 ${COMMIT}

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb1024_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 32 1024 24 ${COMMIT}

#  1n8g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 32 128 24 ${COMMIT}

#  1n8g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb32_gb512_1n8g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 4 true true 32 512 24 ${COMMIT}


## GPT-2

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb1_gb1_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true false 1 1 24 ${COMMIT}

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb64_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 8 64 24 ${COMMIT}

#  1n1g         gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb8_1n1g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 8 8 24 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp4_pp1_mb2_gb2_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true false 2 2 24 ${COMMIT}

#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp4_pp1_mb8_gb64_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 8 64 24 ${COMMIT}

#  1n4g 流水并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb8_gb64_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 8 64 24 ${COMMIT}

#  1n4g 流水并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb12_gb96_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 12 96 24 ${COMMIT}

#  1n4g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp4_mb4_gb32_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 4 32 48 ${COMMIT}

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb1_gb4_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true false 1 4 24 ${COMMIT}

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb32_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 8 32 24 ${COMMIT}

#  1n4g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb128_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 8 128 24 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 模型并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb2_gb2_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true false 2 2 24 ${COMMIT}

#  1n8g 模型并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb8_gb64_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 8 1 true true 8 64 24 ${COMMIT}

#  1n8g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb4_gb64_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 4 64 48 ${COMMIT}

#  1n8g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb6_gb96_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 8 true true 6 96 48 ${COMMIT}

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb1_gb8_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true false 1 8 24 ${COMMIT}

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb64_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 8 64 24 ${COMMIT}

#  1n8g 数据并行        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb512_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 1 true true 8 512 24 ${COMMIT}

#  1n8g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb256_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 8 256 24 ${COMMIT}

#  1n8g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb32_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 1 true true 8 32 24 ${COMMIT}

#  1n8g 数据+流水并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp4_mb8_gb128_1n8g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 1 4 true true 8 128 24 ${COMMIT}
