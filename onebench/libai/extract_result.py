import os
import glob
import argparse

_GLOBAL_ARGS = None


def get_args(extra_args_provider=None):
    global _GLOBAL_ARGS
    if _GLOBAL_ARGS is None:
        _GLOBAL_ARGS = parse_args(extra_args_provider)

    return _GLOBAL_ARGS


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(
        description="OneFlow libai DLPerf Arguments", allow_abbrev=False
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
        default="false_false,false_true,true_false,true_true,master_false",
        help="test commit ,eg: 71de123,55b822e",
    )
    group.add_argument(
        "--test_logs", type=str, default="feat-straighten-op_graph/NVIDIA_GeForce_RTX_3080_Ti", help="log directory"
    )
    group.add_argument("--models_commit", type=str, default="1f10864", help="libai commit, eg: 0bdae6c6")

    group.add_argument(
        "--url_header",
        type=str,
        default="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneAutoTest/onebench/libai",
        help="log directory",
    )

    return parser



def extract_info_from_file(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "iteration:" in line and "time:" in line and "total_throughput:" in line:
                ss = line.split(" ")
                iteration_index = ss.index("iteration:")
                iteration_number = int(ss[iteration_index + 1].strip().split("/")[0])
                time_index = ss.index("time:")                
                samples = float(ss[time_index + 8])
                #samples = float(ss[time_index + 1].strip().split("(")[1][:-1])
                if iteration_number == 219:
                    result_dict["samples"] = samples
            elif "MiB," in line and "utilization" not in line:
                ss = line.split(" ")
                if ss[-1] == 'MiB\n':
                    memory_userd = int(ss[-2])
                    if (
                        "memory" not in result_dict.keys()
                        or result_dict["memory"] < memory_userd
                    ):
                        result_dict["memory"] = memory_userd
                if "nvidia_name" not in result_dict.keys():
                    tmp_nvidia = line.lstrip().split(",")
                    for nvidia_name in tmp_nvidia:
                        if "NVIDIA" in nvidia_name:
                            result_dict["nvidia_name"] = nvidia_name
    return result_dict


def megatron_extract(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "iteration" in line and "tpt:" in line:
                ss = line.split(" ")
                iteration_index = ss.index("iteration")
                iteration_number = int(ss[iteration_index + 6].strip().split("/")[0])
                time_index = ss.index("tpt:")
                samples = float(ss[time_index + 1].strip())
                if iteration_number == 200:
                    result_dict["samples"] = samples
            elif "MiB," in line and "utilization" not in line:
                ss = line.split(" ")
                memory_userd = int(ss[-2])
                if (
                    "memory" not in result_dict.keys()
                    or result_dict["memory"] < memory_userd
                ):
                    result_dict["memory"] = memory_userd
    return result_dict


def extract_result(args, extract_func):
    markdown_table_header = """
| {}                               |"""
    commit_list = args.test_commits.split(",")
    for commit in commit_list:
        # todo if Megatron 替换表头
        markdown_table_header += """ [oneflow@{}](https://github.com/Oneflow-Inc/oneflow/commit/{})  + [libai@{}](https://github.com/Oneflow-Inc/models/commit/{}) |""".format(
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
        
        logs_list = glob.glob(os.path.join(args.test_logs, "{}/*/*/output.log".format(commit)))
        logs_list = sorted(logs_list)

        for log in logs_list:
            result_dict = extract_func(log)
            if "nvidia_name" in result_dict.keys():
                nvidia_name = result_dict["nvidia_name"]
            result_dict["url"] = "{}/{}".format(args.url_header, log)

            tmp_file_path, filename = os.path.split(log)
            file_path = tmp_file_path.split("/")
            case_name = "_".join(file_path[-1].split("_")[1:-2]).lower()
            if file_path[-2] == "55b822e":
                tmp_case = file_path[-1].split("_")[:-2]
                if "bert" in file_path[-1]:
                    model_name = "bert_large_pretrain"
                else:
                    model_name = "gpt2_pretrain"
                mp = int(tmp_case[7][2:])
                pp = int(tmp_case[8][2:])
                node = int(tmp_case[-1].split("n")[0])
                pre_node = int(tmp_case[-1].split("n")[1].split("g")[0])
                dp = int(node * pre_node / mp / pp)
                mbs = int(tmp_case[-3][2:])
                gbs = int(tmp_case[-2][2:])
                acc = int(gbs / mbs / dp)
                case_name = "LibAI_{}_graph_{}_DP{}_{}_zerofalse_stage0_mbs{}_gbs{}_acc{}_{}".format(
                    model_name,
                    "_".join(tmp_case[2:7]),
                    dp,
                    "{}_{}".format(tmp_case[-5],tmp_case[-4]),
                    mbs,
                    gbs,
                    acc,
                    tmp_case[-1],
                )
                case_name = case_name.lower()
                if "mb6_gb6_1n1g" in file_path[-1]:
                    print("+"*4,case_name)
                print("="*10)
                print(case_name)
            else:
                tmp_case = file_path[-1].split("_")
                node = int(tmp_case[-1].split("n")[0])
                pre_node = int(tmp_case[-1].split("n")[1].split("g")[0])
                case_name = "_".join(file_path[-1].split("_")).lower()
                print("*"*10)
                print(case_name)
            result_dict["num_devices_per_node"] = pre_node
            result_dict["num_nodes"] = node
            result_dict["config_url"] = "{}/{}/config.yaml".format(args.url_header,tmp_file_path)
            result_dict["log_url"] = "{}/{}".format(args.url_header,log)
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
        print(case_name)
        for commit in commit_list:
            if commit not in case_value.keys():
                continue
            num_devices_per_node = int(case_value[commit]["num_devices_per_node"])
            num_nodes = int(case_value[commit]["num_nodes"])
            tmp_markdown_table_body = " [{} MiB]({}) / [{} samples/s]({}) ".format(
                case_value[commit]["memory"],
                case_value[commit]["config_url"],
                case_value[commit]["samples"] * num_devices_per_node * num_nodes,
                case_value[commit]["log_url"],
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
