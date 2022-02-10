import os
import json
import re
import matplotlib.pyplot as plt
import time
import requests


def get_pre_merge_commit(pr_number):
    try:
        ob_result = requests.get(
            "https://api.github.com/repos/Oneflow-Inc/oneflow/pulls/{}".format(
                pr_number
            )
        )
        return ob_result.json()["head"]["sha"]
    except:
        return ""


def get_all_file(file_path):
    filelist = []
    for root, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            filelist.append(os.path.join(root, filename))

    return filelist


def extract_result(args):
    logs_list = get_all_file(args.pr_speed_test_log)
    # print(logs_list)
    if len(logs_list) < 1:
        exit(1)
    result_dict = {}
    pr_to_commit = {}
    for txt_result in logs_list:
        test_pr_number = txt_result.split("/")[-4]
        test_commit = txt_result.split("/")[-3]
        if test_pr_number in pr_to_commit.keys():
            pre_merge_commit = pr_to_commit[test_pr_number]
        else:
            pre_merge_commit = get_pre_merge_commit(test_pr_number)
            if pre_merge_commit != "":
                pr_to_commit[test_pr_number] = pre_merge_commit

        if pre_merge_commit != "" and test_commit != pre_merge_commit:
            continue
        if test_pr_number not in result_dict.keys():
            result_dict[test_pr_number] = {}

        test_result = []
        with open(txt_result, "r") as jf:
            for line in jf.readlines():
                for item in line.split("\\n"):
                    tmp_test_result = {}
                    tmp = item.split(" ")
                    if "time:" in item:
                        tmp_test_result["speed_time"] = float(
                            tmp[tmp.index("time:") + 1].split("ms")[0]
                        )
                        tmp_test_result["input_shape"] = "x".join(
                            re.findall(r"input_shape=\[(.*)\]", item)[0].split(", ")
                        )
                        if "world size" in item:
                            tmp_test_result["world_size"] = tmp[-1][-2:-1]
                        else:
                            tmp_test_result["world_size"] = 1

                        if "OneFlow" in item:
                            tmp_test_result["test_name"] = "OneFlow"
                        else:
                            tmp_test_result["test_name"] = "PyTorch"

                        test_result.append(tmp_test_result)

        result_dict[test_pr_number] = test_result

    return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pr_speed_test_log", type=str, required=True)
    parser.add_argument("--git_log_file", type=str, required=True)
    args = parser.parse_args()
    all_pr_speed_result = extract_result(args)

    plot_dict = {}

    with open(args.git_log_file, "r") as jf:
        for line in jf.readlines():
            pr_number = line.split(" ")[-1].strip()[2:-1]
            master_commit = line.split("_")[1]

            if pr_number in all_pr_speed_result.keys():
                for item in all_pr_speed_result[pr_number]:
                    test_case = "{input_shape}_ws{world_size}".format(**item)
                    if test_case not in plot_dict.keys():
                        plot_dict[test_case] = {}
                    if item["test_name"] not in plot_dict[test_case]:
                        plot_dict[test_case][item["test_name"]] = {}
                    plot_dict[test_case][item["test_name"]][master_commit] = item[
                        "speed_time"
                    ]
    speed_test_md = """
- ![{}]({})

"""
    result_markdown = """"""
    for test in plot_dict:
        plt.plot(
            list(plot_dict[test]["OneFlow"].keys())[::-1],
            list(plot_dict[test]["OneFlow"].values())[::-1],
            label="OneFlow",
        )
        plt.plot(
            list(plot_dict[test]["PyTorch"].keys())[::-1],
            list(plot_dict[test]["PyTorch"].values())[::-1],
            color="red",
            label="PyTorch",
        )
        result_markdown += speed_test_md.format(
            test,
            "https://oneflow-test.oss-cn-beijing.aliyuncs.com/oneflow-ci/resnet50/{}/{}.png".format(
                time.strftime("%Y%m%d%H", time.localtime()), test
            ),
        )
        plt.legend(loc="upper right")
        plt.xlabel("master commit")
        plt.ylabel("ms")
        plt.xticks(rotation="90", fontsize=7)
        plt.subplots_adjust(bottom=0.2)
        plt.title("ResNet50" + test)
        plt.savefig(
            "{}/{}/{}.png".format(
                args.pr_speed_test_log, time.strftime("%Y%m%d%H", time.localtime()), test,
            )
        )
        plt.clf()

    with open("{}/ci_speed_test_result.md".format(args.pr_speed_test_log), "w",) as f:
        f.writelines(result_markdown)
