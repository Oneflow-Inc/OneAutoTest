import sys
sys.path.append("/home/xuyongning/OneAutoTest")
from onebench.common.system_info import *
import os
import argparse

def extract_info(dict):

    markdown_template = """
## 硬件信息
- 操作系统: \n\r
    {os_release}
- GPU: `{gpu_name} * {gpu_count}`\n
    driver_version: {driver_version}
    cuda_version: {cuda_version}
- CPU:  运行`lscpu`\n
    Model name: {cpu_name}
    Architecture: {cpu_arch}
    CPU(s): {cpu_core_num}
- 内存: 运行`free -g`或`htop` `{total_mem_size}`
- 多机之间网络: 运行`ibstatus`和`ibstat`\n
    `link layer: {link_layer}`\n
- 磁盘信息:\n
    |      |      |  NAME  |  SIZE  |
    | ---- | ---- | ------ | ------ |\n"""


    body_dict = {}
    body_dict["os_release"] = dict["System"]["os"]
    if isinstance(dict["Accelerator"]["nvidia_info"]["gpu"], list):
        body_dict["gpu_name"] = dict["Accelerator"]["nvidia_info"]["gpu"][0]["product_name"]
    elif isinstance(dict["Accelerator"]["nvidia_info"]["gpu"], dict):
        body_dict["gpu_name"] = dict["Accelerator"]["nvidia_info"]["gpu"]["product_name"]
    body_dict["gpu_count"] = dict["Accelerator"]["gpu_count"]
    body_dict["driver_version"] = dict["Accelerator"]["nvidia_info"]["driver_version"]
    body_dict["cuda_version"] = dict["Accelerator"]["nvidia_info"]["cuda_version"]
    body_dict["cpu_name"] = dict["CPU"]["Model name"]
    body_dict["cpu_arch"] = dict["CPU"]["Architecture"]
    body_dict["cpu_core_num"] = dict["CPU"]["CPU(s)"]
    body_dict["total_mem_size"] = dict["Memory"]["total_capacity"]
    body_dict["link_layer"] = dict["Network"]["ib"]["ib_device_status"][list(dict["Network"]["ib"]["ib_device_status"].keys())[0]]["Port 1:"]["Link layer"]


    markdown_template = markdown_template.format(**body_dict)

    disk_table_body = """   |      |  {disk_type}  |   {disk_name}   |   {disk_size}   |\n"""
    disk_body_dict = {}
    for k, v in dict["Storage"].items():
        if isinstance(v, str):
            continue
        for disk_item in v:
            if "Filesystem" in disk_item:
                disk_body_dict["disk_type"] = "file_system"
                disk_body_dict["disk_name"] = disk_item["Filesystem"]
                disk_body_dict["disk_size"] = disk_item["Size"]
            elif "NAME" in disk_item:
                disk_body_dict["disk_type"] = "block_device"
                disk_body_dict["disk_name"] = disk_item["NAME"]
                disk_body_dict["disk_size"] = disk_item["SIZE"]
            markdown_template += disk_table_body.format(**disk_body_dict)

    return markdown_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-path", type=str, help="md directory")
    args = parser.parse_args()

    systemInfo = SystemInfo()
    sum_dict = systemInfo.get_all()
    markdown_text = extract_info(sum_dict)

    with open("{}".format(args.md_path), "w",) as f:
        f.write(markdown_text)

