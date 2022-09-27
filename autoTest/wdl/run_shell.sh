
# WDL
cd /ssd/oneTest/OneAutoTest/autoTest/wdl/scripts/models && git checkout -f main
cd /ssd/oneTest/OneAutoTest/autoTest/wdl
cp ../../examples/wdl_graph_ddp.sh scripts/models/RecommenderSystems/wide_and_deep/

# wdl-1n1g
cd /ssd/oneTest/OneAutoTest/autoTest/wdl && bash examples/wdl_1n1g.sh

# wdl-1n4g
cd /ssd/oneTest/OneAutoTest/autoTest/wdl && bash examples/wdl_1n4g.sh

# wdl-1n8g
cd /ssd/oneTest/OneAutoTest/autoTest/wdl && bash examples/wdl_1n8g.sh
