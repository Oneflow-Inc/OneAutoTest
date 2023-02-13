import os
import glob
import json
import argparse
from datetime import datetime


_GLOBAL_ARGS = None


def get_args(extra_args_provider=None):
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args(extra_args_provider)

    return _GLOBAL_ARGS


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(
        description="OneFlow ResNet50 DLPerf Arguments", allow_abbrev=False
    )
    parser = _add_training_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def _add_training_args(parser):
    group = parser.add_argument_group(title="training")

    group.add_argument(
        "--test_commits",
        type=str,
        default="master",
        help="test commit ,eg: 65fd4d9,ee07704",
    )
    group.add_argument(
        "--test_logs", type=str, default="./3080TI/ResNet50", help="log directory"
    )
    group.add_argument("--models_commit", type=str, default="fc7cbf8d", help="935e9c1")
    group.add_argument(
        "--url_path",
        type=str,
        default="3080TI/ResNet50",
        help="log directory",
    )
    group.add_argument(
        "--url_header",
        type=str,
        default="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneFlowAutoTest",
        help="log directory",
    )

    return parser


def extract_info_from_file(log_file):
    result_dict = {}
    with open(log_file, "r") as f:
        for line in f.readlines():
            ss = line.lstrip().split(" ")
            if "MiB," in line and "utilization" not in line and len(ss) == 18:

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

            elif ".............." in line and ss[0] in [
                "num_nodes",
                "num_epochs",
                "num_devices_per_node",
                "gpu_num_per_node",  # lazy
                "train_batch_size",
                "use_fp16",
                "graph",
                "ddp",
                "use_gpu_decode",  # ddp graph
                "print_interval",  # ddp graph
            ]:
                result_dict[ss[0]] = ss[-1].strip()
            elif (
                all(
                    tmp_str in line
                    for tmp_str in ["epoch:", "iter:", "throughput:", "top1:"]
                )
                and len(ss) == 15
            ):
                throughtput_key = 0
                iter_key = 0
                epoch_key = 0
                top_key = 0
                ss_key = 0
                for ss_value in ss:
                    if "epoch:" in ss_value:
                        epoch_key = ss_key
                    elif "iter" in ss_value:
                        iter_key = ss_key
                    elif "top1:" in ss_value:
                        top_key = ss_key
                    elif "throughput:" in ss_value:
                        throughtput_key = ss_key
                        break
                    ss_key += 1

                if "[train]" in line:
                    iter_num = ss[iter_key + 1].split("/")[0]
                    throughput_value = float(ss[throughtput_key + 1])
                    if int(iter_num) > 100:
                        if (
                            "max_throughput" not in result_dict.keys()
                            or result_dict["max_throughput"] < throughput_value
                        ):
                            result_dict["max_throughput"] = throughput_value
                        if (
                            "min_throughput" not in result_dict.keys()
                            or result_dict["min_throughput"] > throughput_value
                        ):
                            result_dict["min_throughput"] = throughput_value
                if "[eval]" in line:
                    epoch_num = ss[epoch_key + 1].split("/")[0]
                    top_value = round(float(ss[top_key + 1][:-1]) * 100, 2)
                    if (
                        "max_top1" not in result_dict.keys()
                        or result_dict["max_top1"] < top_value
                    ):
                        result_dict["max_top1"] = top_value
                    if (
                        "min_top1" not in result_dict.keys()
                        or result_dict["min_top1"] > top_value
                    ):
                        result_dict["min_top1"] = top_value
    return result_dict


def extract_result(args, extract_func):
    markdown_table_header = """
| {}                               |"""
    commit_list = args.test_commits.split(",")
    for commit in commit_list:
        markdown_table_header += """ [oneflow@{}](https://github.com/Oneflow-Inc/oneflow/commit/{})  + [models@{}](https://github.com/Oneflow-Inc/models/commit/{}) |""".format(
            commit, commit, args.models_commit, args.models_commit
        )
    markdown_table_header += """
| :-------------------------------------: |"""
    markdown_table_header += (
        """ :----------------------------------------------------------: | """
        * len(commit_list)
    )

    throughput_final_result_dict = {}
    nvidia_name = ""
    for commit in commit_list:
        logs_list = glob.glob(os.path.join(args.test_logs, "{}/*/*.log".format(commit)))
        logs_list = sorted(logs_list)
        for log in logs_list:
            result_dict = extract_func(log)
            if "nvidia_name" in result_dict.keys():
                nvidia_name = result_dict["nvidia_name"]
            result_dict["url"] = "{}/{}".format(args.url_header, log)
            if int(result_dict['num_epochs']) < 20:
                result_dict["max_top1"] = "-"

            file_path, filename = os.path.split(log)
            case_name = "_".join(filename.split("_")[:-3])
            if case_name not in throughput_final_result_dict.keys():
                throughput_final_result_dict[case_name] = {}
            if commit not in throughput_final_result_dict[case_name].keys():
                throughput_final_result_dict[case_name][commit] = result_dict

    markdown_table_header = markdown_table_header.format(nvidia_name)
    last_markdown_body = ""
    for case_name, case_value in throughput_final_result_dict.items():
        markdown_table_body = """
| {} |""".format(
            case_name
        )
        for commit in commit_list:
            num_devices_per_node = int(case_value[commit]["num_devices_per_node"])
            num_nodes = int(case_value[commit]["num_nodes"])
            tmp_markdown_table_body = " [{}-{}] MiB / [{}]({}) / {}".format(
                case_value[commit]["min_memory"],
                case_value[commit]["max_memory"],
                case_value[commit]["max_throughput"] * num_devices_per_node * num_nodes,
                case_value[commit]["url"],
                case_value[commit]["max_top1"],
            )
            tmp_markdown_table_body += " | "
            markdown_table_body += tmp_markdown_table_body
        last_markdown_body += markdown_table_body

    with open(
        "./extract_result.md",
        "w",
    ) as f:
        f.writelines(markdown_table_header)
        f.writelines(last_markdown_body)

    print(markdown_table_header, last_markdown_body)


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
