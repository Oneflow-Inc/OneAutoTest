
MASTER_COMMIT=${1:-"ecafd61b09349a1c6c45333ea6eff96009cf66c0"}
ACC_COMMIT=${2:-"3d5e919cb700d84f52d4cf2730083931f17a91bb"}
BRANCH=${3:-"dev_cc_acc_mem_v5"}

cp args_libai_bert_init.sh args_libai_bert_loss.sh args_libai_gpt2_init.sh args_libai_gpt2_loss.sh args_libai_t5_init.sh args_libai_t5_loss.sh ./libai/tools/
cp ../bert_nl24_nah16_hs1024.py ../gpt2_nl24_nah16_hs1024.py ../t5_nl12_nah12_hs768.py ./libai/configs/
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/bert_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/gpt2_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/t5_nl12_nah12_hs768.py
cp init.sh loss.sh draw_loss.py compose.py ./libai/
cd libai && mkdir loss_txt && mkdir curve

sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 1: \
                      cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                      os.system(cmd)' ./libai/engine/trainer.py
sed -i '/import time/a\import os' ./libai/engine/trainer.py

bash init.sh ${MASTER_COMMIT}

cd libai
sed -i '/if self.iter == 1/if self.iter == 99' ./libai/engine/trainer.py
sed -i '/hooks.PeriodicCheckpointer/#hooks.PeriodicCheckpointer' ./libai/engine/default.py
sed -i '/persistent_workers=True/#persistent_workers=True' ./libai/data/build.py
sed -i '/shuffle=True/shuffle=False' ./libai/data/build.py
sed -i '/total_losses_reduced = sum(metrics_dict.values())/a\            if dist.is_main_process(): \
                txt = open("loss_txt/your_loss.txt", "a") \
                txt.write(str(total_losses_reduced.item())+"\\n")' ./libai/engine/trainer.py
bash loss.sh ${MASTER_COMMIT} ${ACC_COMMIT} ${BRANCH}
