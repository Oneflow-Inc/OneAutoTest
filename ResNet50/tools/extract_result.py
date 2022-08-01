import os
import glob
import json
from datetime import datetime
from config import get_args


def extract_info_from_file(log_file):
    result_dict = {}
    with open(log_file, "r") as f:
        for line in f.readlines():
            ss = line.split(" ")
            if len(ss) == 5 and ss[2] in [
                "num_nodes",
                "num_devices_per_node",  # ddp graph
                "gpu_num_per_node",  # lazy
                "train_batch_size",
                "batch_size_per_device",
                "graph",
                "ddp",
                "use_fp16",
                "use_gpu_decode",  # ddp graph
                "gpu_image_decoder",  # lazy
                "print_interval",  # ddp graph
                "loss_print_every_n_iter",  # lazy
            ]:
                result_dict[ss[2]] = ss[-1].strip()
            elif len(ss) == 3 and ss[0] in [
                "num_nodes",
                "num_devices_per_node",  # ddp graph
                "gpu_num_per_node",  # lazy
                "train_batch_size",
                "batch_size_per_device",
                "graph",
                "ddp",
                "use_fp16",
                "use_gpu_decode",  # ddp graph
                "gpu_image_decoder",  # lazy
                "print_interval",  # ddp graph
                "loss_print_every_n_iter",  # lazy
            ]:
                result_dict[ss[0]] = ss[-1].strip()
            elif "MiB," in line and "utilization" not in line:
                memory_userd = int(ss[-2])
                if (
                    "memory" not in result_dict.keys()
                    or result_dict["memory"] < memory_userd
                ):
                    result_dict["memory"] = memory_userd
            elif all(tmp_str in line for tmp_str in ["loss:", "epoch", "iter", "top"]):
                ss = line.lstrip().split(" ")

                if ("ddp" in result_dict.keys() or "graph" in result_dict.keys()) and (
                    len(ss) == 15 or len(ss) == 17
                ):
                    if "eval" in line:
                        epoch_num = ss[3].split("/")[0]
                        if "eval_epoch_{}_top1".format(epoch_num) in result_dict.keys():
                            result_dict[
                                "eval_epoch_{}_top1".format(epoch_num)
                            ] += float(ss[-6][:-1])
                        else:
                            result_dict["eval_epoch_{}_top1".format(epoch_num)] = float(
                                ss[-6][:-1]
                            )
                    iter_num = ss[5].split("/")[0]
                    # loss = float(ss[7][:-1])
                    result_dict["throughput_{}".format(iter_num)] = float(ss[-4])
                elif len(ss) == 14:
                    iter_num = ss[4][:-1]
                    # loss = float(ss[6][:-1])
                    result_dict["throughput_{}".format(iter_num)] = float(ss[-2])
            elif "validation:" in line:
                epoch_num = ss[2][:-1]
                result_dict["eval_epoch_{}_top1".format(epoch_num)] = float(ss[-6][:-1])

    return result_dict


def compute_throughput(result_dict):
    throughput = 0
    log_interval = int(result_dict["print_interval"])
    if log_interval == 100:
        return result_dict["throughput_200"]
    elif log_interval == 10:
        return result_dict["throughput_20"]
    else:
        for i in range(20, 120, log_interval):
            throughput += result_dict["throughput_{}".format(i)]

        return throughput / 100 / log_interval


def extract_result(args, extract_func):
    logs_list = glob.glob(os.path.join(args.test_log, "*/*.log"))
    logs_list = sorted(logs_list)

    throughput_final_result_dict = {}
    print("## All Results")
    # output Markdown
    if args.run_type == "dlperf":
        markdown_table_header = """

<p align="center">表 {} ResNet50 {}  {} 运行结果</p>

|                                         | [{}](https://github.com/Oneflow-Inc/oneflow/commit/{}) |
| :-------------------------------------: | :----------------------------------------------------------: |
|                                         |                    Memory/Throughput                         |"""
        markdown_table_body = """
| {} | {} MiB / [{}]({}) |"""
    elif args.run_type == "nsys":
        markdown_table_header = """

<p align="center">表 {} ResNet50 {}  {} 运行结果</p>

| [{}](https://github.com/Oneflow-Inc/oneflow/commit/{}) |
| :----------------------------------------------------------: |"""
        markdown_table_body = """
| [{}]({}) |"""
    else:
        markdown_table_header = """

<p align="center">表 {} ResNet50 {}  {} 50E运行结果</p>

|                                         | [{}](https://github.com/Oneflow-Inc/oneflow/commit/{}) |
| :-------------------------------------: | :----------------------------------------------------------: |
|                                         |                    Memory/Throughput/TOP1                    |"""
        markdown_table_body = """
| {} | {} MiB / [{}]({}) / {} |"""

    now = datetime.now()
    tmp_markdown_table_header = markdown_table_header.format(
        now.strftime("%Y-%m-%d"),
        args.model_type,
        args.run_type,
        args.test_commit,
        args.test_commit,
    )
    tmp_markdown_table_body = ""
    case_nodes = ""
    file_path = ""
    for l in logs_list:
        result_dict = extract_func(l)
        case_args = {}
        case_args["model_type"] = args.model_type
        case_args["run_type"] = args.run_type
        case_args["amp_or"] = "FP32"
        if "use_fp16" not in result_dict.keys():
            continue
        if result_dict["use_fp16"] == "True":
            case_args["amp_or"] = "FP16"
        if "train_batch_size" not in result_dict.keys():
            case_args["batch_size"] = result_dict["batch_size_per_device"]
        else:
            case_args["batch_size"] = result_dict["train_batch_size"]

        print(result_dict.keys())
        if "gpu_image_decoder" not in result_dict.keys():
            case_args["use_decode"] = (
                "gpu" if result_dict["use_gpu_decode"] == "True" else "cpu"
            )
        else:
            case_args["use_decode"] = (
                "gpu" if result_dict["gpu_image_decoder"] == "True" else "cpu"
            )
        case_args["node_num"] = result_dict["num_nodes"]
        if "gpu_num_per_node" not in result_dict.keys():
            case_args["pre_gpu_num"] = result_dict["num_devices_per_node"]
        else:
            case_args["pre_gpu_num"] = result_dict["gpu_num_per_node"]
        case_args["node_num"] = result_dict["num_nodes"]

        case_header = "ResNet50_{model_type}_{run_type}_{use_decode}decode_{amp_or}_b{batch_size}_{node_num}n{pre_gpu_num}g".format(
            **case_args
        )
        case_nodes = "{node_num}n{pre_gpu_num}g".format(**case_args)

        if "loss_print_every_n_iter" in result_dict.keys():
            result_dict["print_interval"] = result_dict["loss_print_every_n_iter"]

        file_path, filename = os.path.split(l)
        if (
            any(
                tmp_str in result_dict.keys()
                for tmp_str in [
                    "throughput_100",
                    "throughput_10",
                    "throughput_1",
                    "throughput_1000",
                ]
            )
            and args.run_type != "nsys"
        ):
            throughput = compute_throughput(result_dict)
        elif int(result_dict["num_nodes"]) > 1:
            continue
        else:
            throughput = 0

        if "memory" not in result_dict.keys():
            result_dict["memory"] = "None"

        if case_header not in throughput_final_result_dict.keys():
            throughput_final_result_dict[case_header] = {}
        if args.run_type not in throughput_final_result_dict[case_header].keys():
            throughput_final_result_dict[case_header][args.run_type] = {}
        if (
            case_nodes
            not in throughput_final_result_dict[case_header][args.run_type].keys()
        ):
            throughput_final_result_dict[case_header][args.run_type][case_nodes] = {}

        throughput_final_result_dict[case_header][args.run_type][case_nodes][
            filename
        ] = {}
        throughput_final_result_dict[case_header][args.run_type][case_nodes][filename][
            "url_path"
        ] = args.url_path
        throughput_final_result_dict[case_header][args.run_type][case_nodes][filename][
            "url_header"
        ] = args.url_header
        for key in result_dict:
            throughput_final_result_dict[case_header][args.run_type][case_nodes][
                filename
            ][key] = result_dict[key]
        throughput_final_result_dict[case_header][args.run_type][case_nodes][filename][
            "memory"
        ] = result_dict["memory"]
        throughput_final_result_dict[case_header][args.run_type][case_nodes][filename][
            "throughput"
        ] = throughput
        if args.run_type == "dlperf":
            tmp_markdown_table_body += markdown_table_body.format(
                case_header,
                result_dict["memory"],
                throughput
                if args.model_type == "lazy"
                else (
                    throughput
                    * int(case_args["node_num"])
                    * int(case_args["pre_gpu_num"])
                ),
                "{}{}/{}/{}".format(
                    args.url_header, args.url_path, case_nodes, filename
                ),
            )
        elif args.run_type == "nsys":
            tmp_markdown_table_body += markdown_table_body.format(
                case_header,
                "{}{}/{}/{}.qdrep".format(
                    args.url_header,
                    args.url_path,
                    case_nodes,
                    os.path.splitext(filename)[0],
                ),
            )
        elif args.run_type == "train":
            eval_top = (
                result_dict["eval_epoch_50_top1"] * 100
                if args.model_type == "lazy"
                else float(
                    result_dict["eval_epoch_49_top1"]
                    / (int(case_args["node_num"]) * int(case_args["pre_gpu_num"]))
                    * 100
                )
            )
            throughput_final_result_dict[case_header][args.run_type][case_nodes][
                filename
            ]["top1"] = eval_top
            tmp_markdown_table_body += markdown_table_body.format(
                case_header,
                result_dict["memory"],
                throughput
                if args.model_type == "lazy"
                else (
                    throughput
                    * int(case_args["node_num"])
                    * int(case_args["pre_gpu_num"])
                ),
                "{}{}/{}/{}".format(
                    args.url_header, args.url_path, case_nodes, filename
                ),
                eval_top,
            )

    f = open(
        "{}/ResNet50_{}_{}_{}_result.md".format(
            file_path, args.model_type, args.run_type, case_nodes
        ),
        "w",
    )

    f.writelines(tmp_markdown_table_header)
    f.writelines(tmp_markdown_table_body)
    f.writelines("\r\n")
    f.close()
    # output Markdown
    throughput_final_result_json = json.dumps(throughput_final_result_dict)
    with open(
        "{}/ResNet50_{}_{}_{}_result.json".format(
            file_path, args.model_type, args.run_type, case_nodes
        ),
        "w",
    ) as f:
        f.writelines(throughput_final_result_json)


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
