## 使用方法

参考 https://github.com/Oneflow-Inc/libai/pull/316#issuecomment-1198826336

### 准备环境
``` bash
git clone git@github.com:Oneflow-Inc/OneAutoTest.git
cd OneAutoTest/libai/accuracy_verify
git clone git@github.com:Oneflow-Inc/libai.git
```

### 进行测试
```bash
bash run.sh
```

### 处理测试数据
```bash
jupyter nbconvert --execute --to notebook process_data.ipynb --output new.ipynb
```

## 测试结果

https://github.com/Oneflow-Inc/OneTeam/issues/1670

