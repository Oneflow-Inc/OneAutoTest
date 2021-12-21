import os
import json
from config import get_args


def get_all_file(file_path):
    filelist = []
    for root, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            filelist.append(os.path.join(root, filename))

    return filelist


def get_logs_result(logs_list):
    result_dict = {}
    print(logs_list)
    for json_result in logs_list:
        test_time = json_result.split("/")[1]
        test_commit = json_result.split("/")[2]
        file_name, file_suffix = os.path.splitext(json_result)
        if file_suffix != ".json":
            continue

        with open(json_result, "r") as jf:
            test_result = json.load(jf)

            for test_case in test_result:
                if test_case not in result_dict.keys():
                    result_dict[test_case] = {}

                for test_type in test_result[test_case]:
                    if test_type != "dlperf":
                        continue
                    for node_num in test_result[test_case][test_type]:
                        for log_name in test_result[test_case][test_type][node_num]:
                            result_dict[test_case]["test_time"] = test_time
                            result_dict[test_case]["test_commit"] = test_commit
                            result_dict[test_case]["node_num"] = int(
                                node_num.split("n")[0]
                            )
                            result_dict[test_case]["gpu_pre_node_num"] = int(
                                node_num.split("n")[1].split("g")[0]
                            )
                            if (
                                "latency(ms)"
                                in test_result[test_case][test_type][node_num][
                                    log_name
                                ].keys()
                            ):
                                result_dict[test_case]["throughput"] = round(
                                    test_result[test_case][test_type][node_num][
                                        log_name
                                    ]["latency(ms)"],
                                    3,
                                )
                            else:
                                result_dict[test_case]["throughput"] = round(
                                    test_result[test_case][test_type][node_num][
                                        log_name
                                    ]["throughput"],
                                    3,
                                )
                            result_dict[test_case]["memory"] = test_result[test_case][
                                test_type
                            ][node_num][log_name]["memory"]
                            result_dict[test_case][
                                "oss_url_path"
                            ] = "{}{}/{}/{}".format(
                                test_result[test_case][test_type][node_num][log_name][
                                    "url_header"
                                ],
                                test_result[test_case][test_type][node_num][log_name][
                                    "url_path"
                                ],
                                node_num,
                                log_name,
                            )

    return result_dict


def extract_result(args):
    current_logs_list = get_all_file(args.current_log)
    if len(current_logs_list) < 1:
        print("current log is empty")

        exit(1)
    current_result_dict = get_logs_result(current_logs_list)

    history_logs_list = get_all_file(args.history_log)
    if len(history_logs_list) < 1:
        print("history log is empty")
        exit(1)
    history_result_dict = get_logs_result(history_logs_list)

    markdown_context = """


| Check            | {test_case} | [{current_commit}](https://github.com/Oneflow-Inc/oneflow/commit/{current_commit})({current_test_time}) | [{history_commit}](https://github.com/Oneflow-Inc/oneflow/commit/{history_commit})({history_test_time}) |
| ---------------- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| {throughput_Y_N} | Throughput  | [{current_throughput}]({current_oss_url_path})               | [{history_throughput}]({history_oss_url_path})               |
| {memory_Y_N}     | Memory      | [{current_memory}]({current_oss_url_path}) MiB               | [{history_memory}]({history_oss_url_path}) MiB               |
    


"""
    markdown_args = {}
    result_markdown = """

- OneFlow Master 基于V100上性能测试对比(前一日commit与当天commit的对比)

"""

    for test_case in sorted(current_result_dict):
        if test_case in history_result_dict.keys():
            markdown_args["test_case"] = test_case

            markdown_args["current_test_time"] = current_result_dict[test_case][
                "test_time"
            ]
            markdown_args["current_commit"] = current_result_dict[test_case][
                "test_commit"
            ]

            markdown_args["history_test_time"] = history_result_dict[test_case][
                "test_time"
            ]
            markdown_args["history_commit"] = history_result_dict[test_case][
                "test_commit"
            ]

            markdown_args["current_throughput"] = current_result_dict[test_case][
                "throughput"
            ]
            markdown_args["current_oss_url_path"] = current_result_dict[test_case][
                "oss_url_path"
            ]

            markdown_args["history_throughput"] = history_result_dict[test_case][
                "throughput"
            ]
            markdown_args["history_oss_url_path"] = history_result_dict[test_case][
                "oss_url_path"
            ]

            markdown_args["current_memory"] = current_result_dict[test_case]["memory"]
            markdown_args["history_memory"] = history_result_dict[test_case]["memory"]

            markdown_args["throughput_Y_N"] = (
                "Y"
                if abs(
                    history_result_dict[test_case]["throughput"]
                    - current_result_dict[test_case]["throughput"]
                )
                < args.throughput_interval
                else "N"
            )

            markdown_args["memory_Y_N"] = (
                "Y"
                if current_result_dict[test_case]["memory"] != "None"
                and history_result_dict[test_case]["memory"] != "None"
                and abs(
                    int(current_result_dict[test_case]["memory"])
                    - int(history_result_dict[test_case]["memory"])
                )
                < args.memory_interval
                else "N"
            )
            result_markdown += markdown_context.format(**markdown_args)

    print("success")

    with open("{}/compare_master_dlperf_result.md".format(args.current_log), "w",) as f:
        f.write(result_markdown)

    print("write to file")
    exit(0)


if __name__ == "__main__":
    args = get_args()
    extract_result(args)
