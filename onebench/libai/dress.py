import numpy as np
import math
import matplotlib.pyplot as plt
import json


com_x = []
dev_y_0 = []
dev_y_1 = []
master_y = []
with open("./dev_cc_fuse_nccl_logical/NVIDIA_GeForce_RTX_3080_Ti/5523464/LibAI_bert_large_pretrain_graph_nl24_nah16_hs1024_FP16_actrue_DP8_MP1_PP1_zerotrue_stage2_mbs32_gbs256_acc1_1n8g/metrics.json", "r", encoding="utf-8", newline="\n") as f:
    for line in f.readlines():
        dev_data = json.loads(line)
        com_x.append(dev_data["iteration"])
        dev_y_1.append(dev_data["total_loss"])

with open("./dev_cc_fuse_nccl_logical/NVIDIA_GeForce_RTX_3080_Ti/9ff1e1e7/LibAI_bert_large_pretrain_graph_nl24_nah16_hs1024_FP16_actrue_DP8_MP1_PP1_zerotrue_stage2_mbs32_gbs256_acc1_1n8g/metrics.json", "r", encoding="utf-8", newline="\n") as f:
    for line in f.readlines():
        dev_data = json.loads(line)
        master_y.append(dev_data["total_loss"])

# with open("/home/ouyangyu/workspace/OneAutoTest/onebench/libai/dev_graph_stream_ordered_memory_allocation/NVIDIA_GeForce_RTX_2080_Ti/b65b7c8_0/LibAI_bert_large_pretrain_graph_nl24_nah16_hs1024_FP16_actrue_DP4_MP1_PP1_zerofalse_stage0_mbs32_gbs512_acc4_1n4g/metrics.json", "r", encoding="utf-8", newline="\n") as f:
#     for line in f.readlines():
#         dev_data = json.loads(line)
#         dev_y_0.append(dev_data["total_loss"])


plt.plot(com_x,master_y,label="master")
plt.plot(com_x,dev_y_1,label="fuse_nccl_logical",linestyle = "--")
# plt.plot(com_x,dev_y_1,label="env=1",linestyle = ":")
plt.xlabel("iteration")
plt.ylabel("total_loss")
plt.title('DP8_MP1_PP1_zerotrue_stage2_acc1_1n8g')
plt.legend()   #打上标签
plt.savefig("./LibAI_bert_large_pretrain_graph_nl24_nah16_hs1024_FP16_actrue_DP8_MP1_PP1_zerotrue_stage2_mbs32_gbs256_acc1_1n8g.png")