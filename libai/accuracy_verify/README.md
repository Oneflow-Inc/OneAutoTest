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
bash run.sh "ecafd61b09349a1c6c45333ea6eff96009cf66c0" "3d5e919cb700d84f52d4cf2730083931f17a91bb"
```

### 处理测试数据
注：在将.ipynb转为.pdf之前，需要先安装pandoc和texlive
```bash
jupyter nbconvert --execute --to pdf process_data.ipynb --output new.pdf
```

## 测试结果

https://github.com/Oneflow-Inc/OneTeam/issues/1670

