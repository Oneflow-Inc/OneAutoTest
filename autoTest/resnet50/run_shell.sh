# ResNet50
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50
cp ../../ResNet50/args_train_ddp_graph.sh scripts/models/Vision/classification/image/resnet50/examples

# ResNet50-accuracy
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50/scripts/models && git checkout -f dev_test_resnet50_accuracy
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50 && bash examples/resnet50_graph_ddp_train.sh

# ResNet50-dlperf
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50/scripts/models && git checkout -f dev_test_resnet50_dlperf
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50 && bash examples/resnet50_graph_ddp_dlperf_1n1g.sh
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50 && bash examples/resnet50_graph_ddp_dlperf_1n4g.sh
cd /ssd/oneTest/OneAutoTest/autoTest/resnet50 && bash examples/resnet50_graph_ddp_dlperf_1n8g.sh
