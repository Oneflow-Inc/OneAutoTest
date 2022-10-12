# ResNet50
DATA_PATH="/ssd/dataset/ImageNet/ofrecord"

cp ../../ResNet50/args_train_ddp_graph.sh scripts/models/Vision/classification/image/resnet50/examples

# ResNet50-accuracy
cd scripts/models && git checkout -f dev_test_resnet50_accuracy
cd ../../ && bash examples/resnet50_graph_ddp_train.sh ${DATA_PATH}

# ResNet50-dlperf
cd scripts/models && git checkout -f dev_test_resnet50_dlperf
cd ../../ && bash examples/resnet50_graph_ddp_dlperf_1n1g.sh ${DATA_PATH}
cd ../../ && bash examples/resnet50_graph_ddp_dlperf_1n4g.sh ${DATA_PATH}
cd ../../ && bash examples/resnet50_graph_ddp_dlperf_1n8g.sh ${DATA_PATH}
