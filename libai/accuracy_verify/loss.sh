
MASTER_COMMIT=${1:-"ecafd61b09349a1c6c45333ea6eff96009cf66c0"}
ACC_COMMIT=${2:-"3d5e919cb700d84f52d4cf2730083931f17a91bb"}
BRANCH=${3:-"dev_cc_acc_mem_v5"}
COMMIT=${4:-"master"}

for TEST_COMMIT in ${MASTER_COMMIT} ${ACC_COMMIT}
do
if [ $TEST_COMMIT != $MASTER_COMMIT ];then
    COMMIT = ${ACC_COMMIT:0:7}
    python3 -m pip uninstall oneflow -y
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${BRANCH}/cu112/index.html
else
    python3 -m pip uninstall oneflow -y
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112/${TEST_COMMIT}
fi
pip install -e .

## bert
#  1n1g acc4        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb128_1n1g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 32 128 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g dp4 acc4        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb32_gb512_1n4g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 32 512 ${COMMIT}

#  1n4g 2-D 数据+模型并行 acc4  bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb256_1n4g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 32 256 ${COMMIT}

#  1n4g 2-D 数据+流水并行 acc4  bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp2_mb32_gb256_1n4g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 32 256 ${COMMIT}

#  1n4g 2-D 模型+流水并行 acc4  bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb64_gb256_1n4g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 64 256 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 3-D acc4  bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb64_gb512_1n8g
bash tools/args_libai_bert_loss.sh configs/bert_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 true true 64 512 ${COMMIT}


## gpt
#  1n1g acc4        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb8_gb32_1n1g
#bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 1 0 127.0.0.1 1 1 true true 8 32 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g dp4 acc4        gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb4_gb64_1n4g
bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 4 64 ${COMMIT}

#  1n4g 2-D 数据+模型并行 acc4  gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb8_gb64_1n4g
bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 8 64 ${COMMIT}

#  1n4g 2-D 数据+流水并行 acc4  gpt2_nl24_nah16_hs1024_fp16_actrue_mp1_pp2_mb8_gb64_1n4g
#bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 8 64 ${COMMIT}

#  1n4g 2-D 模型+流水并行 acc4  gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb16_gb64_1n4g
bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 16 64 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 3-D acc4  gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb16_gb128_1n8g
bash tools/args_libai_gpt2_loss.sh configs/gpt2_nl24_nah16_hs1024.py 1 8 0 127.0.0.1 2 2 true true 16 128 ${COMMIT}


## t5
#  1n1g acc4        t5_nl12_nah12_hs768_fp16_actrue_mp1_pp1_mb32_gb128_1n1g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 1 0 127.0.0.1 1 1 true true 32 128 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g dp4 acc4        t5_nl12_nah12_hs768_fp16_actrue_mp1_pp1_mb16_gb256_1n4g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 1 true true 16 256 ${COMMIT}

#  1n4g 2-D 数据+模型并行 acc4  t5_nl12_nah12_hs768_fp16_actrue_mp2_pp1_mb128_gb1024_1n4g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 1 true true 128 1024 ${COMMIT}

#  1n4g 2-D 数据+流水并行 acc4  t5_nl12_nah12_hs768_fp16_actrue_mp1_pp2_mb32_gb256_1n4g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 2 true true 32 256 ${COMMIT}

#  1n4g 2-D 模型+流水并行 acc4  t5_nl12_nah12_hs768_fp16_actrue_mp2_pp2_mb64_gb256_1n4g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 2 true true 64 256 ${COMMIT}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#  1n8g 3-D acc4  t5_nl12_nah12_hs768_fp16_actrue_mp2_pp2_mb64_gb512_1n8g
bash tools/args_libai_t5_loss.sh configs/t5_nl12_nah12_hs768.py 1 8 0 127.0.0.1 2 2 true true 64 512 ${COMMIT}
done
