
### 运行测试
```bash
git clone https://github.com/Oneflow-Inc/OneAutoTest.git
cd OneAutoTest/libai/auto_parallel
git clone https://github.com/Oneflow-Inc/libai.git
cd libai && pip install -e . && cd ..
mkdir libai_dataset && cd libai_dataset
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx
cd .. && vim run.sh # 修改L7 L8中的ImageNet路径
bash run.sh
```

### 处理数据
先更改 `extract_auto_parallel.py` L87,L95中的日期

`python3 extract_auto_parallel.py --test-log /path/to/bert/`
