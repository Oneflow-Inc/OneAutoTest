
# cd /ssd/oneTest/OneAutoTest/autoTest/
# git clone git@github.com:Oneflow-Inc/libai.git
# git clone git@github.com:Oneflow-Inc/models.git

#python3 -m pip uninstall -y oneflow && python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/refs/heads/master/cu112/index.html && python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# WDL
cd /ssd/oneTest/OneAutoTest/autoTest
cp ../examples/wdl_graph_ddp.sh scripts/models/RecommenderSystems/wide_and_deep/

# wdl-1n1g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n1g.sh

# wdl-1n4g
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n4g.sh


# libai
'''
cd /ssd/oneTest/OneAutoTest/autoTest
cp ../libai/args_libai_bert.sh scripts/libai/tools/
cp ../libai/bert_nl24_nah16_hs1024.py scripts/libai/configs
sed -i 's/\/path\/to/\/data\/dataset\/libai_dataset/g' scripts/libai/configs/bert_nl24_nah16_hs1024.py
sed -i '$a train.rdma_enabled = False' scripts/libai/configs/bert_nl24_nah16_hs1024.py

cp ../libai/args_libai_gpt2.sh scripts/libai/tools/
cp ../libai/gpt2_nl24_nah16_hs1024.py scripts/libai/configs
sed -i 's/\/path\/to/\/data\/dataset\/libai_dataset/g' scripts/libai/configs/gpt2_nl24_nah16_hs1024.py
sed -i '$a train.rdma_enabled = False' scripts/libai/configs/gpt2_nl24_nah16_hs1024.py

# libai-bert-1n1g-1n4g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n1g.sh

# libai-gpt-1n1g-1n4g
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n1g.sh
'''
