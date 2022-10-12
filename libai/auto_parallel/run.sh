
cp args_libai_bert.sh args_libai_gpt2.sh args_libai_t5.sh args_libai_vit.sh args_libai_swin.sh ./libai/tools/
cp ../bert_nl24_nah16_hs1024.py ../gpt2_nl24_nah16_hs1024.py ../t5_nl12_nah12_hs768.py ../vit_imagenet.py ../swin_imagenet.py ./libai/configs/
sed -i 's/\/path\/to/..\/libai_dataset/g' ./libai/configs/bert_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/..\/libai_dataset/g' ./libai/configs/gpt2_nl24_nah16_hs1024.py
sed -i 's/\/path\/to/..\/libai_dataset/g' ./libai/configs/t5_nl12_nah12_hs768.py
sed -i 's/\/path\/to\/dataset/\/ssd\/dataset\/ImageNet\/extract/g' ./libai/configs/vit_imagenet.py
sed -i 's/\/path\/to\/dataset/\/ssd\/dataset\/ImageNet\/extract/g' ./libai/configs/swin_imagenet.py
sed -i '$agraph.auto_parallel.mainstem_algo = True \
graph.auto_parallel.enable_auto_parallel_ignore_user_sbp_config = True \
graph.auto_parallel.sbp_collector = False' ./libai/configs/bert_nl24_nah16_hs1024.py
sed -i '$agraph.auto_parallel.mainstem_algo = True \
graph.auto_parallel.enable_auto_parallel_ignore_user_sbp_config = True \
graph.auto_parallel.sbp_collector = False' ./libai/configs/gpt2_nl24_nah16_hs1024.py
sed -i '$agraph.auto_parallel.mainstem_algo = True \
graph.auto_parallel.enable_auto_parallel_ignore_user_sbp_config = True \
graph.auto_parallel.sbp_collector = False' ./libai/configs/t5_nl12_nah12_hs768.py
sed -i '$agraph.auto_parallel.mainstem_algo = True \
graph.auto_parallel.enable_auto_parallel_ignore_user_sbp_config = True \
graph.auto_parallel.sbp_collector = False' ./libai/configs/vit_imagenet.py
sed -i '$agraph.auto_parallel.mainstem_algo = True \
graph.auto_parallel.enable_auto_parallel_ignore_user_sbp_config = True \
graph.auto_parallel.sbp_collector = False' ./libai/configs/swin_imagenet.py


cd libai
sed -i '/for self.iter in range(start_iter, max_iter):/a\                    if self.iter == 99: \
                        cmd = "nvidia-smi --query-gpu=timestamp,name,driver_version,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv" \
                        os.system(cmd)' ./libai/engine/trainer.py
sed -i '/import time/a\import os' ./libai/engine/trainer.py
sed -i '/hooks.PeriodicCheckpointer/#hooks.PeriodicCheckpointer' ./libai/engine/default.py

cp ../case_withoutAcc.sh ../case_withAcc.sh ./
bash case_withoutAcc.sh
bash case_withAcc.sh
