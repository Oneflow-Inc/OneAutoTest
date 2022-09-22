from system_info import *
import os

if __name__ == "__main__":
    systemInfo = SystemInfo()
    os_info = systemInfo._run_cmd('cat /etc/os-release')
    nvidia_info = systemInfo._run_cmd('nvidia-smi')
    lscpu_info = systemInfo._run_cmd('lscpu')
    mem_info = systemInfo._run_cmd('free -g')
    disk_info = systemInfo._run_cmd('sudo fdisk -l')
    ibstatus_info = systemInfo._run_cmd('ibstatus')
    ibstat_info = systemInfo._run_cmd('ibstat')
    sum_dict = systemInfo.get_all()


    markdown_text = """## 硬件信息 \n"""
    markdown_text += """- 操作系统: \n运行`cat /etc/os-release`或`head -n 1 /etc/issue`\n```\n{}```\n""".format(os_info)

    if isinstance(sum_dict["Accelerator"]["nvidia_info"]["gpu"], list):
        markdown_text += """- GPU: `{} * {}`\n\r\n\r""".format(sum_dict["Accelerator"]["nvidia_info"]["gpu"][0]["product_name"], sum_dict["Accelerator"]["nvidia_info"]["attached_gpus"])
    elif isinstance(sum_dict["Accelerator"]["nvidia_info"]["gpu"], dict):
        markdown_text += """- GPU: `{} * {}`\n\r\n\r""".format(sum_dict["Accelerator"]["nvidia_info"]["gpu"]["product_name"], sum_dict["Accelerator"]["nvidia_info"]["attached_gpus"])
    markdown_text += """运行`nvidia-smi`,可输出GPU显卡信息, `Driver Version: {}`, `CUDA Version: {}`\n```\n{}```\n""".format(sum_dict["Accelerator"]["nvidia_info"]["driver_version"], sum_dict["Accelerator"]["nvidia_info"]["cuda_version"], nvidia_info)
    markdown_text += """运行`nvidia-smi topo -m`\n```\n {} ```\n""".format(sum_dict["Accelerator"]["topo"])

    markdown_text += """- CPU: `{}` {}核\n 运行`lscpu`:\n```\n{}```\n""".format(sum_dict["CPU"]["Model name"], sum_dict["CPU"]["CPU(s)"], lscpu_info)

    markdown_text += """- 内存:`{}`\n运行`free -g`或`htop`\n```\n{}```\n""".format(sum_dict["Memory"]["total_capacity"], mem_info)

    markdown_text += """- 磁盘信息:\n运行`sudo fdisk -l`:\n```\n{}```\n""".format(disk_info)

    markdown_text += """- 多机之间网络:\n运行`ibstatus`\n```\n{}```\n运行`ibstat`查看，`link layer: {}`\n```\n{}```\n""".format(ibstatus_info, sum_dict["Network"]["ib"]["ib_device_status"][list(sum_dict["Network"]["ib"]["ib_device_status"].keys())[0]]["Port 1:"]["Link layer"], ibstat_info)

    with open("report.md", "w",) as f:
        f.write(markdown_text)
