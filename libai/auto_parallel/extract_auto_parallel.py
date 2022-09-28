import os
import glob
import yaml
from extract_config import get_args


def extract_info_from_file(log_file):
    result_dict = {}
    result_dict["samples"] = 0
    result_dict["memory"] = 0
    with open(log_file, "r") as f:
        for line in f.readlines():
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
    return result_dict


def get_config(yaml_file):
    with open("{}/config.yaml".format(yaml_file), "r") as f:
        config_data = yaml.full_load(f)

    return config_data



def extract_result(args, extract_func):

    logs_list = glob.glob(os.path.join(args.test_log, "*/output.log"))
    logs_list = sorted(logs_list)

    logs_id = [log[:log.index("ap")] for log in logs_list]
    logs_id = list(set(logs_id))

    throughput_final_result_dict = {}
    case_dict = {}
    markdown_table_header = """

|      | 自动并行 关 | 自动并行 开|
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |"""

    markdown_table_body_sider = """
| {case_name}   |"""

    markdown_table_body = """ [{libai_memory}]({libai_yaml}) MiB/[{libai_samples}]({libai_log}) samples/s                                |"""
    
    markdown_line = {}

    for l in logs_list:
        ap_start_point = l.index("ap")
        l_ap = l[ap_start_point:].split("_")[0]

        parallel_start_point = l.index("mp")
        l_parallel = l[parallel_start_point:parallel_start_point + 7]
        
        libai_result_dict = extract_func(l)
        tmp_file_name = l.split("/")
        case_header = "_".join(tmp_file_name[-2].split("_")[1:-2]).lower()
        case_header_ap_start_point = case_header.index("ap")
        case_header_ap_end_point = case_header[case_header_ap_start_point:].index("_") + case_header_ap_start_point
        case_header = case_header[:case_header_ap_start_point] + case_header[case_header_ap_end_point + 1:]

        if case_header not in throughput_final_result_dict.keys():
            throughput_final_result_dict[case_header] = {}
            case_dict[case_header] = {}

        case_dict[case_header]["case_name"] = case_header
        throughput_final_result_dict[case_header]["libai_memory"] = libai_result_dict[
            "memory"
        ]
        throughput_final_result_dict[case_header][
            "libai_yaml"
        ] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/AutoParallel/0827/{}/config.yaml".format(
            "/".join(tmp_file_name[-3:-1])
        )
        throughput_final_result_dict[case_header]["libai_samples"] = libai_result_dict[
            "samples"
        ]
        throughput_final_result_dict[case_header][
            "libai_log"
        ] = "https://oneflow-test.oss-cn-beijing.aliyuncs.com/AutoParallel/0827/{}".format(
            "/".join(tmp_file_name[-3:])
        )
        
        if l_parallel not in markdown_line.keys():
            markdown_line[l_parallel] = markdown_table_body_sider.format(
                **case_dict[case_header]
            )
        markdown_line[l_parallel] += markdown_table_body.format(
            **throughput_final_result_dict[case_header]
        )


    with open("{}/dlperf_result.md".format(args.test_log), "w",) as f:
        f.write(markdown_table_header)
        f.write(markdown_line["mp1_pp1"])
        f.write(markdown_line["mp4_pp1"])
        f.write(markdown_line["mp2_pp1"])
        f.write(markdown_line["mp2_pp2"])
        f.write(markdown_line["mp1_pp4"])
        f.write(markdown_line["mp1_pp2"])


if __name__ == "__main__":
    args = get_args()
    extract_result(args, extract_info_from_file)
