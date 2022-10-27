## 使用方法

参考 https://github.com/Oneflow-Inc/libai/pull/316#issuecomment-1198826336

### 准备环境
``` bash
git clone https://github.com/Oneflow-Inc/OneAutoTest.git
cd OneAutoTest/libai/accuracy_verify && git checkout dev_libai_accuracy
git clone https://github.com/Oneflow-Inc/libai.git
cd libai && pip install -e . && cd ..
mkdir libai_dataset && cd libai_dataset
wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/bert-base-chinese-vocab.txt && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-merges.txt && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/gpt2-vocab.json && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.bin && wget https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest/libai/dataset/loss_compara_content_sentence.idx
cd ..
```

### 进行测试
```bash
bash run.sh "ecafd61b09349a1c6c45333ea6eff96009cf66c0" "3d5e919cb700d84f52d4cf2730083931f17a91bb" "dev_cc_acc_mem_v5"
```

### 处理测试数据
- 在linux服务器上配置jupyter notebook

  `pip install jupyter`

  `ipython`

  依次输入 `from notebook.auth import passwd` 和 `passwd()` 保存该密钥，exit退出

  `jupyter notebook --generate-config`

  `vim ~/.jupyter/jupyter_notebook_config.py` 添加配置
  ```
  # Nginx访问时会出现跨域访问，需要在这里允许
  c.NotebookApp.allow_origin = '*'
  
  # 禁止随意修改密码
  c.NotebookApp.allow_password_change = False
  
  # 是否允许远程访问
  c.NotebookApp.allow_remote_access = True
  
  # IP
  c.NotebookApp.ip = 'localhost'
  
  # 端口
  c.NotebookApp.port = 9820
  
  # 工作目录
  c.NotebookApp.notebook_dir = '/home/xuyongning'
  
  # 启动Jupyter Notebook之后是否打开浏览器
  c.NotebookApp.open_browser = False
  
  # 客户端打开Jupyter Notebook的密码哈希值
  c.NotebookApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$L0z+zk1RjIB83lV0fNeo7A$ILwIDki4PKGNdH4/2Yik8O2MoTk1KzNM/QYOUCd+D+w'
  ```
  最后一行改为刚才保存的密钥

  远端服务器输入 `jupyter notebook`，本地浏览器输入 `localhost:9820` 即可访问并修改服务器上的notebook文件


- 运行notebook并输出pdf

  注：在将.ipynb转为.pdf之前，需要先安装pandoc和texlive

  - 下载pandoc：https://github.com/jgm/pandoc/releases/tag/2.19.2

    `export PATH="/path/to/pandoc-2.19.2/bin:$PATH"`

  - 下载texlive
    ```
    wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
    zcat install-tl-unx.tar.gz | tar xf -
    cd install-tl-*
    vim install-tl # 安装路径'/usr/local/texlive'修改成自己的安装路径
    perl ./install-tl --no-interaction
    ```
    `export PATH="/path/to/texlive/2022/bin/x86_64-linux:$PATH"`
  
  - 修改notebook中个别的参数，0.0.4显存表格处的commit，0.0.5中ROW和COLUMN的值。修改extract_libai_libai.py中oss的地址

  - 运行
    ```bash
    # 如果notebook中有中文，则先运行并输出tex文件，再在tex中添加中文宏包，最后将tex文件转为pdf文件
    jupyter nbconvert --execute --to latex process_data.ipynb --output new.tex
    vim new.tex && 在其中添加一行 \usepackage{ctex} 用于解决中文输出问题
    xelatex new.tex

    # 如果notebook中没有中文，直接一步运行下方语句即可
    jupyter nbconvert --execute --to pdf process_data.ipynb --output new.pdf
    ```
    运行后耐心等待几分钟，会生成pdf文件

## 测试结果

https://github.com/Oneflow-Inc/OneTeam/issues/1670

