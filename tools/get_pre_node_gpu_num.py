import os


def get_pre_node_gpu_num():
    gpu_result = os.popen("nvidia-smi -L")
    res = gpu_result.read()
    gpu_num = len(res.splitlines())
    print(gpu_num)


if __name__ == "__main__":
    get_pre_node_gpu_num()
