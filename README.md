# OneAutoTest
Auto-Test System

# OneFlow安装

- ### conda环境
`wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh && bash Miniconda3-py38_4.12.0-Linux-x86_64.sh`

### OneFlow选择cuda安装

- ### 分支安装
```
# 其他分支，替换master即可
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112
```
- ### 基于某个commit安装
```
# 替换commit hash即可
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/commit/2d080aac5c41c02346641a5576b359bc95399214/cu112/index.html
```