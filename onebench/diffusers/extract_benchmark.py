import os
import click
import os
import glob
import numpy as np


def extract_info_from_file(log_file):
    result_dict = {}
    result_dict["iter_sec"] = []
    with open(log_file, "r") as f:
        for line in f.readlines():
            ss = line.lstrip().split(" ")
            if "MiB," in line and "utilization" not in line:
                memory_userd = int(ss[-2])
                if (
                    "max_memory" not in result_dict.keys()
                    or result_dict["max_memory"] < memory_userd
                ):
                    result_dict["max_memory"] = memory_userd
                if (
                    "min_memory" not in result_dict.keys()
                    or result_dict["min_memory"] > memory_userd
                ):
                    result_dict["min_memory"] = memory_userd

                if "nvidia_name" not in result_dict.keys():
                    tmp_nvidia = line.lstrip().split(",")
                    for nvidia_name in tmp_nvidia:
                        if "NVIDIA" in nvidia_name:
                            result_dict["nvidia_name"] = nvidia_name
            elif "100%" in line and "it/s" in line and "50/50" in line:
                ss = line.lstrip().split("timestamp,")[0]
                ss = ss.lstrip().split(" ")
                it_seconds = float(ss[-1].strip()[:-1].split("it/s")[0])
                result_dict["iter_sec"].append(it_seconds)
                if (
                    "max_iter_sec" not in result_dict.keys()
                    or result_dict["max_iter_sec"] < it_seconds
                ):
                    result_dict["max_iter_sec"] = it_seconds
                    result_dict["max_iter_sec_str"] = ss[-1].strip()[:-1]
                if (
                    "min_iter_sec" not in result_dict.keys()
                    or result_dict["min_iter_sec"] > it_seconds
                ):
                    result_dict["min_iter_sec_str"] = ss[-1].strip()[:-1]
                    result_dict["min_iter_sec"] = it_seconds
            if "RuntimeError:" in line:
                result_dict["max_iter_sec"] = "-"
                result_dict["max_iter_sec_str"] = "-"
                result_dict["min_iter_sec"] = "-"
                result_dict["min_iter_sec_str"] = "-"
                result_dict["max_memory"] = "OOM"
                result_dict["min_memory"] = "OOM"
                return result_dict
    return result_dict



@click.command()
@click.option("--oneflow_commit", default="151eccef")
@click.option("--dl_frame", default="oneflow,pytorch")
@click.option("--test_logs", default="diffusion-benchmark-0.9.0", help="nvidia_name/dl_frame/commit/*.log")
@click.option("--url_header", default="https://oneflow-test.oss-cn-beijing.aliyuncs.com/StableDiffusion")
def extract_benchmark(oneflow_commit, dl_frame, test_logs, url_header):
    logs_list = glob.glob(os.path.join(test_logs, "*/*/*/*.log"))
    logs_list = sorted(logs_list)
    markdown_table_header = """
| Device                     | Case              | OneFlow([master@{}](https://github.com/Oneflow-Inc/oneflow/commit/{})) + diffusers([oneflow-fork](https://github.com/Oneflow-Inc/diffusers)) | Pytorch({}) + diffusers([main](https://github.com/huggingface/diffusers)) |
| -------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |"""
    final_result_dict = {}
    dl_frame_list = dl_frame.split(",")
    for log in logs_list:
        file_path, filename = os.path.split(log)
        # if "oneflow" in filename and oneflow_commit not in filename:
        #     print("_"*10)
        #     continue
        result_dict = extract_info_from_file(log)
        result_dict["url"] = "{}/{}".format(url_header, log)

        for frame in dl_frame_list:

            if frame in filename:
                case_name_device = filename.split("_{}_".format(frame))[0]
                case_name_sd = filename.split("_{}_".format(frame))[1].split(".py")[0]

                if case_name_device not in final_result_dict.keys():
                    final_result_dict[case_name_device] = {}
                if case_name_sd not in final_result_dict[case_name_device].keys():
                    final_result_dict[case_name_device][case_name_sd] = {}
                if frame not in final_result_dict[case_name_device][case_name_sd].keys():
                    final_result_dict[case_name_device][case_name_sd][frame] = {}
                final_result_dict[case_name_device][case_name_sd][frame] = result_dict
                

    markdown_table_header = markdown_table_header.format(
        oneflow_commit, oneflow_commit, "1.13.0a0+d0d6b1f"
    )
    last_markdown_body = ""
    for case_name_device, case_value in final_result_dict.items():
        for case_name_sd_key,case_name_sd_value in case_value.items():
            markdown_table_body = """
    | {}      | {} | [{} 、{} MiB]({}) | [{} 、{} MiB]({}) |""".format(
                case_name_device,
                case_name_sd_key,
                np.median(case_name_sd_value["oneflow"]["iter_sec"]),
                case_name_sd_value["oneflow"]["max_memory"],
                case_name_sd_value["oneflow"]["url"].replace("+","%2B"),
                np.median(case_name_sd_value["pytorch"]["iter_sec"]),
                case_name_sd_value["pytorch"]["max_memory"],
                case_name_sd_value["pytorch"]["url"].replace("+","%2B"),
            )
            last_markdown_body += markdown_table_body

    with open(
        "./extract_benchmark.md",
        "w",
    ) as f:
        f.writelines(markdown_table_header)
        f.writelines(last_markdown_body)

    print(markdown_table_header, last_markdown_body)


if __name__ == "__main__":
    extract_benchmark()
