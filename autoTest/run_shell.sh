
# cd /ssd/oneTest/OneAutoTest/autoTest/
# git clone git@github.com:Oneflow-Inc/libai.git
# git clone git@github.com:Oneflow-Inc/models.git

#python3 -m pip uninstall -y oneflow && python3 -m pip install oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/canary/refs/heads/master/cu112/index.html && python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ResNet50
#cd /ssd/oneTest/OneAutoTest/autoTest
#cp ../ResNet50/args_train_ddp_graph.sh scripts/models/Vision/classification/image/resnet50/examples

# ResNet50-accuracy
#cd /ssd/oneTest/OneAutoTest/autoTest/scripts/models && git checkout -f dev_test_resnet50_accuracy
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/resnet50_graph_ddp_train.sh

# ResNet50-dlperf
#cd /ssd/oneTest/OneAutoTest/autoTest/scripts/models && git checkout -f dev_test_resnet50_dlperf
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/resnet50_graph_ddp_dlperf_1n1g.sh
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/resnet50_graph_ddp_dlperf_1n2g.sh
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/resnet50_graph_ddp_dlperf_1n4g.sh
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/resnet50_graph_ddp_dlperf_1n8g.sh



# WDL
#cd /ssd/oneTest/OneAutoTest/autoTest/scripts/models && git checkout -f main
#cd /ssd/oneTest/OneAutoTest/autoTest
#cp ../examples/wdl_graph_ddp.sh scripts/models/RecommenderSystems/wide_and_deep/

# wdl-1n1g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n1g.sh

# wdl-1n2g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n2g.sh

# wdl-1n4g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n4g.sh

# wdl-1n8g
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/wdl_1n8g.sh


# libai

#cd /ssd/oneTest/OneAutoTest/autoTest
#cp ../libai/args_libai_bert.sh scripts/libai/tools/
#cp ../libai/bert_nl24_nah16_hs1024.py scripts/libai/configs
#sed -i 's/01b1d32/oneflow-28/g' scripts/libai/tools/args_libai_bert.sh
#sed -i 's/RUN_COMMIT/HOSTNAME/g' scripts/libai/tools/args_libai_bert.sh
#sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' scripts/libai/configs/bert_nl24_nah16_hs1024.py
#sed -i '$a train.rdma_enabled = False' scripts/libai/configs/bert_nl24_nah16_hs1024.py

#cp ../libai/args_libai_gpt2.sh scripts/libai/tools/
#cp ../libai/gpt2_nl24_nah16_hs1024.py scripts/libai/configs
#sed -i 's/01b1d32/oneflow-28/g' scripts/libai/tools/args_libai_gpt2.sh
#sed -i 's/RUN_COMMIT/HOSTNAME/g' scripts/libai/tools/args_libai_gpt2.sh
#sed -i 's/\/path\/to/\/ssd\/dataset\/libai_dataset/g' scripts/libai/configs/gpt2_nl24_nah16_hs1024.py
#sed -i '$a train.rdma_enabled = False' scripts/libai/configs/gpt2_nl24_nah16_hs1024.py

# libai-bert
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n1g.sh
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n2g.sh
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n4g.sh
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_bert_1n8g.sh

# libai-gpt
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n1g.sh
#cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n2g.sh
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n4g.sh
cd /ssd/oneTest/OneAutoTest/autoTest && bash examples/libai_gpt_1n8g.sh

