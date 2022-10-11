
### 运行测试
```bash
git clone https://github.com/Oneflow-Inc/OneAutoTest.git
cd OneAutoTest/libai/auto_parallel
git clone https://github.com/Oneflow-Inc/libai.git
cd libai && pip install -e . && cd ..
bash run.sh
```

### 处理数据
先更改 `extract_auto_parallel.py` L87,L95中的日期

`python3 extract_auto_parallel.py --test-log /path/to/bert/`
