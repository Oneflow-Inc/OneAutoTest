
# cd /ssd/oneTest/OneAutoTest/autoTest/
# git clone https://github.com/Oneflow-Inc/libai.git
# git clone https://github.com/Oneflow-Inc/models.git

#python3 -m pip uninstall -y oneflow && python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/refs/heads/master/cu112/index.html

# libai
'''
cd /ssd/oneTest/OneAutoTest/autoTest
cp ../libai/args_libai_bert.sh scripts/libai/tools/
cp ../libai/bert_nl24_nah16_hs1024.py scripts/libai/configs
sed -i 's/\/path\/to/\/data\/dataset\/libai_dataset/g' scripts/libai/configs/bert_nl24_nah16_hs1024.py

cp ../libai/args_libai_gpt2.sh scripts/libai/tools/
cp ../libai/gpt2_nl24_nah16_hs1024.py scripts/libai/configs
sed -i 's/\/path\/to/\/data\/dataset\/libai_dataset/g' scripts/libai/configs/gpt2_nl24_nah16_hs1024.py
'''
# libai-bert-1n1g-1n4g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n1g.sh

# libai-gpt-1n1g-1n4g
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n1g.sh
