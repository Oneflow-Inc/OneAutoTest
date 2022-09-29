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
bash run.sh "ecafd61b09349a1c6c45333ea6eff96009cf66c0" "3d5e919cb700d84f52d4cf2730083931f17a91bb" "dev_cc_acc_mem_v5"
```

### 处理测试数据
注：在将.ipynb转为.pdf之前，需要先安装pandoc和texlive

- 下载pandoc：https://github.com/jgm/pandoc/releases/tag/2.19.2

    `export PATH="/path/to/pandoc-2.19.2/bin:$PATH"`

- 下载texlive
    ```
    wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    zcat install-tl-unx.tar.gz | tar xf -
    cd install-tl-*
    vim install-tl # 修改安装路径'/usr/local/texlive'
    perl ./install-tl --no-interaction
    ```
    `export PATH="/path/to/texlive/2022/bin/x86_64-linux:$PATH"`

- 运行
    ```bash
    jupyter nbconvert --execute --to latex process_data.ipynb --output new.tex
    vim new.tex && 在其中添加一行 \usepackage{ctex} 用于解决中文输出问题
    xelatex new.tex

    # 如果notebook中没有中文，直接运行下方语句即可
    jupyter nbconvert --execute --to pdf process_data.ipynb --output new.pdf
    ```
    运行后耐心等待几分钟，会生成pdf文件

## 测试结果

https://github.com/Oneflow-Inc/OneTeam/issues/1670

