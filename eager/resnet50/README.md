## resnet50 eager测试

### 1. 准备测试环境

`docker run --rm -it -v --shm-size=16g --ulimit memlock=-1 --privileged --name eager_resnet50 --net host -v /home/user_name:/workspace nvcr.io/nvidia/pytorch:21.07-py3`

### 2. 克隆仓库

```bash
mkdir eager_resnet50
cd eager_resnet50
git clone git@github.com:Oneflow-Inc/models.git
```

### 3. 复制脚本

将resnet50_test.sh args_test_speed.sh 和 process_speed_data.py 拷贝至eager_resnet50路径下

并将args_test_speed.sh中的/path/to/修改为自己的绝对路径

### 4. 运行

- 运行之前需要先确认

    - 4个环境变量——在resnet50_test.sh中修改
    - 测试用例（batch size & ddp卡数）——在resnet50_test.sh中修改
    - 重复实验的次数——在args_test_speed.sh中修改
    - 是否跑nsys——在args_test_speed.sh中修改

- 确认完后，运行`bash resnet50_test.sh 43074d24166c5ff51c485667dd14e408ded04d5d`
  
  第一个参数需要改成测试commit的全称

### 5. 处理结果数据，生成表格

- 运行时间数据：保存在data/路径下，命名为test_eager_commit_*_

  nsys文件：保存在data/路径下，命名为resnet50_eager_*.qdrep

  将data文件夹上传至oss

- 修改process_speed_data.py中的 /path/to/路径 和 nsys_root

  运行`python3 process_speed_data.py --test_commits 43074d24166c5ff51c485667dd14e408ded04d5d`

  将会在当前路径下生成一个process_res文件，存储markdown格式的表格
