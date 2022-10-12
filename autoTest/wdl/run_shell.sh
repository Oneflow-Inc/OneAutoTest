
# WDL
DATA_PATH="/data/dataset/wdl_ofrecord"

cd scripts/models && git checkout -f main && cd ../../
cp wdl_graph_ddp.sh scripts/models/RecommenderSystems/wide_and_deep/

# wdl-1n1g
bash examples/wdl_1n1g.sh ${DATA_PATH}

# wdl-1n4g
bash examples/wdl_1n4g.sh ${DATA_PATH}

# wdl-1n8g
bash examples/wdl_1n8g.sh ${DATA_PATH}
