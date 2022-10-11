import os
import glob
import yaml
from extract_config import get_args


def extract_info_from_file(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0

    #result_dict["ori_cost_max"] = 0
    result_dict["update_cost_max"] = 0
    result_dict["cur_size_max"] = 0
    result_dict["lower_bound_max"] = 0

    update_cost_max = 0
    cur_size_max = 0
    lower_bound_max = 0

    #result_dict["ori_cost_sum"] = 0
    result_dict["update_cost_sum"] = 0
    result_dict["cur_size_sum"] = 0
    result_dict["lower_bound_sum"] = 0

    result_dict["update_cost_ratio"] = 0
    result_dict["cur_size_ratio"] = 0
    result_dict["lower_bound_ratio"] = 0

    flag = 0
    cur_cache = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
            if flag == 1:
                if "cost" in line:
                    flag = 0
                else:
                    result_dict["update_cost_sum"] += cur_cache
                    flag = 0
            if "iteration:" in line and "time:" in line:
                ss = line.split(" ")
                iteration_index = ss.index("iteration:")
                iteration_number = int(ss[iteration_index + 1].strip().split("/")[0])
                time_index = ss.index("time:")
                samples = float(ss[time_index + 8])
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
            elif "cost" in line:
                ss = line.split(" ")
                #ori = int(ss[2].split(',')[0])
                update = int(ss[5].split('\\')[0])
                #result_dict["ori_cost_sum"] += ori
                result_dict["update_cost_sum"] += update
                if update > update_cost_max:
                    update_cost_max = update
                flag = 0
            elif "bound" in line:
                ss = line.split(" ")
                cur = int(ss[2].split(',')[0])
                lower = int(ss[6].split('\\')[0])
                result_dict["cur_size_sum"] += cur
                result_dict["lower_bound_sum"] += lower
                if cur > cur_size_max:
                    cur_size_max = cur
                if lower > lower_bound_max:
                    lower_bound_max = lower
                flag = 1
                cur_cache = cur

        if result_dict["lower_bound_sum"] != 0:
            result_dict["update_cost_ratio"] = round((result_dict["update_cost_sum"] - result_dict["lower_bound_sum"]) * 100 / result_dict["lower_bound_sum"], 2)
            result_dict["cur_size_ratio"] = round((result_dict["cur_size_sum"] - result_dict["lower_bound_sum"]) * 100 / result_dict["lower_bound_sum"], 2)
        result_dict["lower_bound_ratio"] = 0

        result_dict["update_cost_max"] = update_cost_max
        result_dict["cur_size_max"] = cur_size_max
        result_dict["lower_bound_max"] = lower_bound_max

    return result_dict


def get_config(yaml_file):
    with open("{}/config.yaml".format(yaml_file), "r") as f:
        config_data = yaml.full_load(f)

    return config_data



def extract_result(args, extract_func):

    logs_list = glob.glob(os.path.join(args.test_log, "*/output.log"))
    logs_list = sorted(logs_list)

    logs_id = [log[:log.index("al")] for log in logs_list]
    logs_id = list(set(logs_id))

    throughput_final_result_dict = {}
    case_dict = {}
    markdown_table_header = """

| 拉直 关 | current size | lower bound | updated cost | 拉直 开| current size | lower bound | updated cost | 测试用例名称 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |"""

    markdown_table_body_sider = """ {case_name}   |"""

    markdown_table_body = """ [{libai_memory}]({libai_yaml}) MiB/[{libai_samples}]({libai_log}) samples/s | sum {libai_cur_size_sum} / ratio {libai_cur_size_ratio}% | sum {libai_lower_bound_sum} / ratio {libai_lower_bound_ratio}% | sum {libai_update_cost_sum} / ratio {libai_update_cost_ratio}%    |"""
    
    markdown_line = {}

    for l in logs_list:
        al_start_point = l.index("al")
        l_al = l[al_start_point:].split("_")[0]

        parallel_start_point = l.index("mp")
        l_parallel = l[parallel_start_point:parallel_start_point + 7] + "_" + l.split("_")[-3]
        
        libai_result_dict = extract_func(l)
        tmp_file_name = l.split("/")
        case_header = "_".join(tmp_file_name[-2].split("_")[1:-2]).lower()
        if "alt" in case_header:
            case_header_al_start_point = case_header.index("alt")
        else:
            case_header_al_start_point = case_header.index("alf")
        case_header_al_end_point = case_header[case_header_al_start_point:].index("_") + case_header_al_start_point
        case_header = case_header[:case_header_al_start_point] + case_header[case_header_al_end_point + 1:]

        if case_header not in throughput_final_result_dict.keys():
            throughput_final_result_dict[case_header] = {}
            case_dict[case_header] = {}

        case_dict[case_header]["case_name"] = case_header
        throughput_final_result_dict[case_header]["libai_memory"] = libai_result_dict[
            "memory"
        ]
        throughput_final_result_dict[case_header][
            "libai_yaml"
        ] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/straighten_algorithm/0926/{}/config.yaml".format(
            "/".join(tmp_file_name[-3:-1])
        )
        throughput_final_result_dict[case_header]["libai_samples"] = libai_result_dict[
            "samples"
        ]
        throughput_final_result_dict[case_header][
            "libai_log"
        ] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/straighten_algorithm/0926/{}".format(
            "/".join(tmp_file_name[-3:])
        )

        throughput_final_result_dict[case_header]["libai_cur_size_sum"] = libai_result_dict["cur_size_sum"]
        throughput_final_result_dict[case_header]["libai_lower_bound_sum"] = libai_result_dict["lower_bound_sum"]
        throughput_final_result_dict[case_header]["libai_update_cost_sum"] = libai_result_dict["update_cost_sum"]

        throughput_final_result_dict[case_header]["libai_cur_size_max"] = libai_result_dict["cur_size_max"]
        throughput_final_result_dict[case_header]["libai_lower_bound_max"] = libai_result_dict["lower_bound_max"]
        throughput_final_result_dict[case_header]["libai_update_cost_max"] = libai_result_dict["update_cost_max"]

        throughput_final_result_dict[case_header]["libai_cur_size_ratio"] = libai_result_dict["cur_size_ratio"]
        throughput_final_result_dict[case_header]["libai_update_cost_ratio"] = libai_result_dict["update_cost_ratio"]
        throughput_final_result_dict[case_header]["libai_lower_bound_ratio"] = libai_result_dict["lower_bound_ratio"]


        if l_parallel not in markdown_line.keys():
            markdown_line[l_parallel] = """
| """
            markdown_line[l_parallel] += markdown_table_body.format(
                **throughput_final_result_dict[case_header]
            )
        else:
            markdown_line[l_parallel] += markdown_table_body.format(
                **throughput_final_result_dict[case_header]
            )
            markdown_line[l_parallel] += markdown_table_body_sider.format(
                **case_dict[case_header]
            )

    with open("{}/dlperf_result.md".format(args.test_log), "w",) as f:
        f.write(markdown_table_header)
        f.write(markdown_line["mp1_pp1_1n1g"])
        f.write(markdown_line["mp1_pp1_1n4g"])
        f.write(markdown_line["mp4_pp1_1n4g"])
        f.write(markdown_line["mp2_pp1_1n4g"])
        f.write(markdown_line["mp2_pp2_1n4g"])
        f.write(markdown_line["mp1_pp4_1n4g"])
        f.write(markdown_line["mp1_pp2_1n4g"])


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
