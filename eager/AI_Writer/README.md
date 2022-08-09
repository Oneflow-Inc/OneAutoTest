## AI-Writer eager测试

### 1. 准备测试环境

torch1.9.1+cu111

[CUDA Toolkit 11.1.0 | NVIDIA Developer](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal)

```bash
export PATH="/path/to/cuda-11.1/bin:$PATH" # 需要将path/to/修改成本地cuda安装路径
export LD_LIBRARY_PATH="/path/to/cuda-11.1/lib64:$LD_LIBRARY_PATH"
export PATH="/path/to/cuda-11.1/nsight-systems-2020.3.4/bin:$PATH"
source ~/.bashrc
```

`pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

### 2. 克隆仓库

```bash
mkdir writer
cd writer
git clone git@github.com:zhongshsh/AI-Writer-Web.git
git clone git@github.com:BlinkDL/AI-Writer.git
mv AI-Writer-Web/ AI_Writer_Web/
mv AI-Writer/ AI_Writer/
```

克隆后，将[AI-Writer-Web/infer.py](https://github.com/xyn1201/AI-Writer-Web/blob/main/infer.py) 和 [AI-Writer/infer.py](https://github.com/xyn1201/AI-Writer/blob/main/infer.py) 两个infer.py文件拷贝到两个仓库对应的文件夹下，并修改其中的/path/to/绝对路径名

再将[AI-Writer-Web/config.py](https://github.com/xyn1201/AI-Writer-Web/blob/main/config.py) 拷贝到两个仓库对应的文件夹下

### 3. 复制脚本

将OneAutoTest/eager/AI_Writer下的4个脚本拷贝到writer路径下