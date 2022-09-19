
cp args_libai_bert_init.sh args_libai_bert_loss.sh args_libai_gpt2_init.sh args_libai_gpt2_loss.sh args_libai_t5_init.sh args_libai_t5_loss.sh ./libai/tools/
cp ../bert_nl24_nah16_hs1024.py ../gpt2_nl24_nah16_hs1024.py ../t5_nl12_nah12_hs768.py ./libai/configs/
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/bert_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/gpt2_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' ./libai/configs/t5_nl12_nah12_hs768.py
cp init.sh loss.sh draw_loss.py compose.py ./libai/
cd libai && mkdir loss_txt && mkdir curve

sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 1:\n \
                      cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"\n \
                      os.system(cmd)' ./libai/engine/trainer.py
sed -i '/import time/a\import os' ./libai/engine/trainer.py

bash init.sh

sed -i '/99/1' ./libai/engine/trainer.py
sed -i '/hooks.PeriodicCheckpointer/#hooks.PeriodicCheckpointer' ./libai/engine/default.py
sed -i '/persistent_workers=True/#persistent_workers=True' ./libai/data/build.py
sed -i '/shuffle=True/shuffle=False' ./libai/data/build.py
sed -i '/total_losses_reduced = sum(metrics_dict.values())/a\            if dist.is_main_process():\n \
                txt = open("loss_txt/your_loss.txt", "a")\n \
                txt.write(str(total_losses_reduced.item())+"\n")' ./libai/engine/trainer.py
bash loss.sh
