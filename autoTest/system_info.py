import subprocess
import os
import re
import psutil
import socket
from pathlib import Path
import argparse
import sys
import time


_TWO_20 = float(2 ** 20)


def get_os():
    output = subprocess.check_output(["cat", "/etc/os-release",])
    return output.decode("utf-8").strip()


def get_cpu_memory(pid):
    try:
        process = psutil.Process(pid)
        meminfo_attr = (
            "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        )
        return getattr(process, meminfo_attr)()[0] / _TWO_20
    except:
        return False


def gpu_topo():
    output = subprocess.check_output(["nvidia-smi", "topo", "-m",])
    return output.decode("utf-8").strip()


def gpu_info():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv",]
    )
    return output.decode("utf-8").strip()


def gpu_memory_used():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    return output.decode("utf-8").strip()


def get_gpu_memory_used_by_pid(output, current_pid):
    if not output:
        return ""
    for line in output.split("\n"):
        pid, mem_used = map(int, re.split(",? ", line)[:2])
        print(pid)
        print(current_pid)
        if pid == current_pid:
            return mem_used

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, for GPU training, this is recommended to be set to the number of GPUs in your system so that each process can be bound to a single GPU.",
    )
    parser.add_argument("--interval-time", type=int, default=30, help="seconds")
    args = parser.parse_args()

    host_name = socket.gethostname()
    exit_file = Path("{}/{}".format(args.log_dir, host_name))

    while not exit_file.is_dir():
        time.sleep(20)

    with open("{}/machine.info".format(exit_file), "a",) as f:
        f.writelines("\n{}gpu_info{}\n".format("*" * 8, "*" * 8))
        f.writelines(gpu_info())
        f.writelines("\n{}gpu_topo{}\n".format("*" * 8, "*" * 8))
        f.writelines(gpu_topo())
        f.writelines("\n{}os_info{}\n".format("*" * 8, "*" * 8))
        f.writelines(get_os())

    
    while True:
        gpu_output = gpu_memory_used()
        kill_flag = 0
        for local_rank in range(0, args.nproc_per_node):
            file_path = "{}/local_rank_{}/{}".format(
                args.log_dir, local_rank, host_name
            )
            pid = -1
            realpath = ""
            for root, dirnames, filenames in os.walk(file_path):
                date_time_str = ""
                for filename in filenames:
                    filename_split = filename.split(".")
                    if (
                        len(filename_split) > 2
                        and filename_split[-2] > date_time_str
                        and filename.split(".")[-1].isdigit()
                    ):
                        date_time_str = filename_split[-2]
                        pid = int(filename.split(".")[-1])
                        realpath = os.path.join(root, filename)
            if pid < 0:
                kill_flag += 1
                continue
            print(gpu_output)
            gpu_memory = get_gpu_memory_used_by_pid(gpu_output, pid)
            cpu_memory = get_cpu_memory(pid)
            if not cpu_memory:
                kill_flag += 1
            else:
                with open("{}.memory".format(realpath), "a",) as f:
                    f.writelines(
                        "timestamp:{},cpu_memory:{}\n".format(time.time(), cpu_memory)
                    )
                    f.writelines(
                        "timestamp:{},gpu_memory:{}\n".format(time.time(), gpu_memory)
                    )

        if kill_flag == args.nproc_per_node:
            sys.exit(1)

        time.sleep(args.interval_time)
