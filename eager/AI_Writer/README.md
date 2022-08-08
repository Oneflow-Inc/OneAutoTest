## AI-Writer 测试

### 1. 准备测试环境

torch1.9.1+cu111

### 2. 克隆仓库

```bash
mkdir writer
cd writer
git clone git@github.com:zhongshsh/AI-Writer-Web.git
git clone git@github.com:BlinkDL/AI-Writer.git
```

克隆后，将[AI-Writer-Web/infer.py at main · xyn1201/AI-Writer-Web (github.com)](https://github.com/xyn1201/AI-Writer-Web/blob/main/infer.py) 和 [AI-Writer/infer.py at main · xyn1201/AI-Writer (github.com)](https://github.com/xyn1201/AI-Writer/blob/main/infer.py) 两个infer.py文件拷贝到两个仓库对应的文件夹下，并修改其中的/path/to/绝对路径名

### 3. 复制脚本

将 [OneAutoTest/eager/AI_Writer at dev_eager_test · Oneflow-Inc/OneAutoTest (github.com)](https://github.com/Oneflow-Inc/OneAutoTest/tree/dev_eager_test/eager/AI_Writer) 中的4个脚本拷贝到writer路径下
