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
        description="OneFlow occl DLPerf Arguments", allow_abbrev=False
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
        "--test_frame",
        type=str,
        default="occl,nccl",
        help="test commit ,eg: OCCL,NCCL",
    )
    group.add_argument(
        "--test_logs", type=str, default="20230307", help="log directory"
    )
    group.add_argument("--models_commit", type=str, default="1f10864", help="libai commit, eg: 0bdae6c6")

    group.add_argument(
        "--url_header",
        type=str,
        default="https://oneflow-test.oss-cn-beijing.aliyuncs.com/OneAutoTest/onebench/occl",
        help="log directory",
    )

    return parser



def extract_info_from_file(log_file,test_frame,case_name):
    result_dict = {}
    result_dict["size"] = 0
    result_dict["time"] = 0
    result_dict["algbw"] = 0
    result_dict["busbw"] = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
            ss = line.split(" ")

            if "#" not in line and "testlog" not in line and len(ss) > 6:
                ss = [x for x in ss if x != '']
                print(ss)
                result_dict["size"] = ss[0]
                i = 2
                if test_frame == "occl":
                    i = 0
                    if case_name == "reduce_scatter" or case_name == "broadcast":
                        i = 1
                    elif case_name == "reduce":
                        i = 2

                result_dict["time"] = ss[3+i]
                result_dict["algbw"] = ss[4+i]
                result_dict["busbw"] = ss[5+i]
    return result_dict


def extract_result(args, extract_func):
    logs_list = glob.glob(os.path.join(args.test_logs, "*.log"))
    logs_list = sorted(logs_list)
    finally_result = {}
    for log in logs_list:
        
        test_case_list = log.split("_nccl_")
        test_frame = "nccl"
        if len(test_case_list) < 2:
            test_case_list = log.split("_ofccl_")
            test_frame = "occl"
        gpu_name = test_case_list[0]
        test_case_list = test_case_list[1].split("_perf_")
        case_name = test_case_list[0]
        case_size = test_case_list[1].split("_")[0]
        gpu_number = test_case_list[1].split("_")[1].split(".")[0]
        test_case_name = "{}_{}_{}_{}".format(gpu_name,case_name,gpu_number,case_size)
        result_dict = extract_func(log,test_frame,case_name)

        result_dict["gpu_name"] = gpu_name
        result_dict["log_url"] = "{}/{}".format(args.url_header, log)

        if test_case_name not in finally_result.keys():
            finally_result[test_case_name] = {}
        if test_frame not in finally_result[test_case_name].keys():
            finally_result[test_case_name][test_frame] = {}
        finally_result[test_case_name][test_frame] = result_dict
    
    markdown_table_header = """
| Case(gpu_name_case_gpu_number)                               |"""
    commit_list = args.test_frame.split(",")
    for commit in commit_list:
        markdown_table_header += """ {}[time(us)/algbw(GB/s)] |""".format(commit)
    markdown_table_header += """
| :-------------------------------------: |"""
    markdown_table_header += (
        """ :----------------------------------------------------------: | """
        * len(commit_list)
    )

    last_markdown_body = ""
    for case_name, case_value in finally_result.items():
        markdown_table_body = """
| {} |""".format(
            case_name
        )
        for frame in commit_list:
            tmp_markdown_table_body = " [{} / {}]({}) ".format(
                case_value[frame]["time"],
                case_value[frame]["algbw"],
                case_value[frame]["log_url"],
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

    #print(markdown_table_header, last_markdown_body)


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
