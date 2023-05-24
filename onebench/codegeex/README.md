使用说明：

- 在A100机器上运行
- 在正常terminal中运行bash initialize_docker.sh
- 在打开的container bash中运行 bash original_script.sh
- 确保environment.yml文件存在

示例输出，保存在results_table.md中：

| L | OneFlow[Mem(MiB)/Time(s)] | PyTorch[Mem(MiB)/Time(s)] | fastertransformer[Mem(MiB)/Time(s)] |
| --- | --- | --- | --- |
| 128 | 25687/0.039 | 26137/0.056 | 26892/2.832 |
| 256 | 25987/3.035 | 26231/4.364 | 26892/5.421 |
| 512 | 26707/9.158 | 27194/9.934 | 26892/11.236 |
| 1024 | 27763/21.968 | 28654/24.382 | 28932/25.541 |
| 2048 | 33093/50.033 | 34028/58.842 | 30294/56.203 |