
### Prepare for testing
- install oneflow
```bash
# modify ${MASTER_COMMIT} to your test commit
python3 -m pip uninstall oneflow -y
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112/${MASTER_COMMIT}
```

- clone repo
```bash
git clone https://github.com/Oneflow-Inc/OneAutoTest.git
cd OneAutoTest && git checkout dev_autoTest && cd autoTest
```
- change dataset dir in
  - `libai/run_shell.sh` L7 L13
  - `resnet50/run_shell.sh` DATA_PATH
  - `wdl/run_shell.sh` DATA_PATH

### Test
```bash
# Test libai
cd libai && mkdir scripts && cd scripts
git clone https://github.com/Oneflow-Inc/libai.git
cd .. && bash run_shell.sh && cd ..

# Test resnet50
cd resnet50 && mkdir scripts && cd scripts
git clone https://github.com/Oneflow-Inc/models.git
cd .. && bash run_shell.sh && cd ..

# Test wdl
cd wdl && mkdir scripts && cd scripts
git clone https://github.com/Oneflow-Inc/models.git
cd .. && bash run_shell.sh && cd ..
```

