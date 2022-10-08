export CUDA_VISIBLE_DEVICES=0,1,4,5

## bert 开Acc

#  1n4g 模型并行        bert_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb128_gb1024_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 32 256

#  1n4g 流水并行        bert_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb64_gb1024_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 32 256 false 48

#  1n4g 数据并行        bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp1_mb128_gb8192_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 8 256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 8 256 true

#  1n4g 数据+模型并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb128_gb4096_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 16 256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 16 256 true

#  1n4g 3D并行   bert_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb128_gb2048_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 32 256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 32 256 true

#  1n4g 数据+流水并行   bert_nl24_nah16_hs1024_fp16_actrue_mp1_pp2_mb128_gb2048_1n4g
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 16 256
bash tools/args_libai_bert.sh configs/bert_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 16 256 true



## gpt  开Acc

#  1n4g 模型并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 4 1 true true 8 64

#  1n4g 流水并行        gpt2_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 4 true true 8 64 false 48

#  1n4g 数据并行        gpt2_nl24_nah16_hs1024_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 2 64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 1 true true 2 64 true

#  1n4g 数据+模型并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 4 64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 1 true true 4 64 true

#  1n4g 3D并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 8 64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 2 2 true true 8 64 true

#  1n4g 数据+流水并行   gpt2_nl24_nah16_hs1024_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 4 64
bash tools/args_libai_gpt2.sh configs/gpt2_nl24_nah16_hs1024.py 1 4 0 127.0.0.1 1 2 true true 4 64 true


sed -i 's/flow.boxing.nccl.enable_use_compute_stream(True)/flow.boxing.nccl.enable_use_compute_stream(False)/g' libai/models/utils/graph_base.py

## t5  开Acc

#  1n4g 模型并行        t5_nl24_nah16_hs1024_fp16_actrue_mp8_pp1_mb32_gb256_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 4 1 true true 16 128

#  1n4g 流水并行        t5_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 4 true true 16 128 false 24

#  1n4g 数据并行        t5_nl12_nah12_hs768_fp16_actrue_mp1_pp1_mb32_gb2048_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 1 true true 4 128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 1 true true 4 128 true

#  1n4g 数据+模型并行   t5_nl12_nah12_hs768_fp16_actrue_mp2_pp1_mb32_gb1024_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 1 true true 8 128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 1 true true 8 128 true

#  1n4g 3D并行   t5_nl12_nah12_hs768_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 2 true true 16 128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 2 2 true true 16 128 true

#  1n4g 数据+流水并行   t5_nl12_nah12_hs768_fp16_actrue_mp1_pp2_mb32_gb512_1n4g
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 2 true true 8 128
bash tools/args_libai_t5.sh configs/t5_nl12_nah12_hs768.py 1 4 0 127.0.0.1 1 2 true true 8 128 true

sed -i 's/flow.boxing.nccl.enable_use_compute_stream(False)/flow.boxing.nccl.enable_use_compute_stream(True)/g' libai/models/utils/graph_base.py


## vit  开Acc

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g 模型并行        vit_imagenet_fp16_acfalse_mp8_pp1_mb8_gb8_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 4 1 true true 128 1024

#  1n4g 流水并行        vit_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 4 true true 128 1024

#  1n4g 数据并行        vit_imagenet_fp16_acfalse_mp1_pp1_mb4_gb32_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 1 true true 32 1024
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 1 true true 32 1024 true

#  1n4g 数据+模型并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 1 true true 64 1024
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 1 true true 64 1024 true

#  1n4g 3D并行   vit_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 2 true true 128 1024
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 2 2 true true 128 1024 true

#  1n4g 数据+流水并行   vit_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 2 true true 64 1024
bash tools/args_libai_vit.sh configs/vit_imagenet.py 1 4 0 127.0.0.1 1 2 true true 64 1024 true



## swin  开Acc

export CUDA_VISIBLE_DEVICES=0,1,4,5
#  1n4g 模型并行        swin_imagenet_fp16_actrue_mp8_pp1_mb8_gb8_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 4 1 true true 128 1024

#  1n4g 流水并行        swin_nl48_nah16_hs1024_fp16_actrue_mp1_pp8_mb16_gb256_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 4 true true 128 1024

#  1n4g 数据并行        swin_imagenet_fp16_actrue_mp1_pp1_mb4_gb32_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 1 true true 32 1024
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 1 true true 32 1024 true

#  1n4g 数据+模型并行   swin_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 1 true true 64 1024
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 1 true true 64 1024 true

#  1n4g 3D并行   swin_imagenet_fp16_actrue_mp2_pp2_mb32_gb512_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 2 true true 128 1024
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 2 2 true true 128 1024 true

#  1n4g 数据+流水并行   swin_imagenet_fp16_actrue_mp2_pp1_mb32_gb128_1n4g
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 2 true true 64 1024
bash tools/args_libai_swin.sh configs/swin_imagenet.py 1 4 0 127.0.0.1 1 2 true true 64 1024 true
