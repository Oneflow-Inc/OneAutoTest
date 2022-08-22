import os

gpu_result = os.popen("nvidia-smi -L")
res = gpu_result.read()
gpu_num = len(res.splitlines())
print(gpu_num)

