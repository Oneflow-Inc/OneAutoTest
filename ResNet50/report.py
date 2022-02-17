import os
import json
from config import get_args


def get_all_file(file_path):
    filelist = []
    for root, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            filelist.append(os.path.join(root, filename))

    return filelist


def extract_result(args):
    logs_list = get_all_file(args.test_log)
    if len(logs_list) < 1:
        exit(1)

    result_dict = {}
    test_time = logs_list[0].split("/")[1]
    test_commit = logs_list[0].split("/")[2]
    markdown_table_header = """
    
- ###  {} ResNet50 基于[{}](https://github.com/Oneflow-Inc/oneflow/commit/{}) 在V100上的性能测试结果

| Check            | Case Name  | Graph/Lazy                                                   |
| ---------------- | ---------- | ------------------------------------------------------------ |"""
    result_markdown = markdown_table_header.format(test_time, test_commit, test_commit)
    markdown_case_body = """
|                  | {}         |                                                              |"""
    markdown_table_body = """
| {throughput_Y_N} | Throughput | [{graph_throughput}]({graph_url_path}) / [{lazy_throughput}]({lazy_url_path}) {throughput_graph_lazy} |
| {memory_Y_N}     | Memory     | [{graph_memory}]({graph_url_path}) MiB / [{lazy_memory}]({lazy_url_path}) MiB {memory_graph_lazy} |"""

    check_header_n = """ """
    for json_result in logs_list:
        with open(json_result, "r") as jf:
            print(jf)
            test_result = json.load(jf)

            for test_case in test_result:
                tmp_test_case = test_case.split("_dlperf_")
                if len(tmp_test_case) != 2:
                    continue

                if tmp_test_case[1] not in result_dict.keys():
                    result_dict[tmp_test_case[1]] = {}

                tmp_run_type = tmp_test_case[0].split("_")[1]

                for test_type in test_result[test_case]:
                    if test_type != "dlperf":
                        continue
                    for node_num in test_result[test_case][test_type]:
                        node_nums = int(node_num.split("n")[0])
                        gpu_pre_node_num = int(node_num.split("n")[1].split("g")[0])
                        for log_name in test_result[test_case][test_type][node_num]:
                            tmp_throughput = test_result[test_case][test_type][
                                node_num
                            ][log_name]["throughput"]
                            if tmp_run_type == "graph":
                                tmp_throughput = (
                                    tmp_throughput * node_nums * gpu_pre_node_num
                                )
                            result_dict[tmp_test_case[1]][
                                "{}_throughput".format(tmp_run_type)
                            ] = tmp_throughput
                            result_dict[tmp_test_case[1]][
                                "{}_memory".format(tmp_run_type)
                            ] = test_result[test_case][test_type][node_num][log_name][
                                "memory"
                            ]
                            result_dict[tmp_test_case[1]][
                                "{}_url_path".format(tmp_run_type)
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
                            result_dict[tmp_test_case[1]][
                                "rank".format(tmp_run_type)
                            ] = (node_nums * gpu_pre_node_num)

    for test in result_dict:
        if (
            "lazy_throughput" not in result_dict[test]
            or "graph_throughput" not in result_dict[test]
        ):
            continue

        result_markdown += markdown_case_body.format(test)
        tmp_result = result_dict[test]
        tmp_result["throughput_Y_N"] = (
            "Y"
            if tmp_result["lazy_throughput"] - tmp_result["graph_throughput"]
            < args.throughput_interval
            else "N"
        )

        tmp_result["memory_Y_N"] = (
            "Y"
            if tmp_result["lazy_memory"] != "None"
            and tmp_result["graph_memory"] != "None"
            and int(tmp_result["graph_memory"]) - int(tmp_result["lazy_memory"])
            < args.memory_interval
            else "N"
        )
        ratio = round(
            (tmp_result["graph_throughput"] / tmp_result["lazy_throughput"]) * 100, 3,
        )
        tmp_result["throughput_graph_lazy"] = (
            "≈ {} %".format(
                round(
                    (tmp_result["graph_throughput"] / tmp_result["lazy_throughput"])
                    * 100,
                    3,
                )
            )
            if tmp_result["graph_throughput"] != 0
            and tmp_result["lazy_throughput"] != 0
            else ""
        )
        if (tmp_result["rank"] > 4 and ratio < 85) or (
            tmp_result["rank"] <= 4 and ratio < 96
        ):
            check_header_n = """
- ### ❌ check fail

"""
        tmp_result["memory_graph_lazy"] = (
            "≈ {} % ".format(
                round(
                    (int(tmp_result["graph_memory"]) / int(tmp_result["lazy_memory"]))
                    * 100,
                    3,
                )
            )
            if tmp_result["lazy_memory"] != "None"
            and tmp_result["graph_memory"] != "None"
            else ""
        )
        result_markdown += markdown_table_body.format(**tmp_result)

    with open("{}/ResNet50_dlperf_result.md".format(args.test_log), "w",) as f:
        f.writelines(check_header_n + result_markdown)


if __name__ == "__main__":
    args = get_args()
    extract_result(args)
