
### 运行测试
```bash
git clone git@github.com:Oneflow-Inc/OneAutoTest.git
cd OneAutoTest/libai/straighten_algorithm
git clone git@github.com:Oneflow-Inc/libai.git
cd libai && pip install -e . && cd ..
bash run.sh
```

### 处理数据
`python3 extract_straighten.py --test-log /path/to/bert/`
