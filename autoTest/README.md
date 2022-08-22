
### Prepare for testing
```bash
git clone git@github.com:Oneflow-Inc/OneAutoTest.git
cd OneAutoTest && git checkout dev_autoTest
cd autoTest && mkdir scripts && cd scripts
git clone git@github.com:Oneflow-Inc/libai.git
git clone git@github.com:Oneflow-Inc/models.git
cd .. && mkdir oss && cd oss # and install ossutil64, set ossutilconfig
cd ..

```

### Test
```bash
# one terminal runs
bash run_shell.sh

# another terminal runs
bash info_shell.sh
```
