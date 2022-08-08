import numpy as np
import copy
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_commits", type=str)

    args = parser.parse_args()

    file_suffixes = args.test_commits.split(" ")

    with open("process_res", "w") as process_res:
        header = "| 是否开启分配与计算分离 /  <br />是否开启tmp_compute / <br />是否开启stream_wait / <br />是否开启Infer cache |"
        format_line = "| ------------------------------------------------------------ |"
        for suffix in file_suffixes:
            header += (" OneFlow  [%s](https://github.com/Oneflow-Inc/oneflow/commit/%s)  / <br />Relative speed |  PyTorch  time  |" % (suffix[:7], suffix))
            format_line += " ------------------------------------------------------------ | ------------- |"
        process_res.write(header + "\n")
        process_res.write(format_line + "\n")

        mean_times = []
        mean_torch_times = []
        mean_relative_speed = []

        root_oneflow = "/path/to/writer/data/oneflow"
        root_pytorch = "/path/to/writer/data/pytorch"
        dict = {}
        for file in os.listdir(root_oneflow):
            if file.startswith("test_eager_commit"):
                if file not in dict.keys():
                    dict[file] = {}

                with open(os.path.join(root_oneflow, file), "r", encoding="utf8") as f:
                    count_i = 0                    
                    exec_time = []
                    lines = copy.deepcopy(list(f.readlines()))
                    lines_copy = copy.deepcopy(lines)
                    for i, line in enumerate(lines):
                        if "OneFlow AI-Writer time: " in line: 
                            count_i += 1
                            if count_i <= 10:
                                continue
                            if not line.startswith("OneFlow AI-Writer time"):
                                line = line[line.find("OneFlow AI-Writer time"):]
                            
                            # if 1n1d or not
                            if "output_preLen=" in line:
                                if '1n1d' not in dict[file].keys():
                                    dict[file]['1n1d'] = {}

                                # catch length
                                len_start_pos = line.find('=') + 1
                                len_size = line[len_start_pos:-1]

                                if len_size not in dict[file]['1n1d'].keys():
                                    dict[file]['1n1d'][len_size] = {}
                                    dict[file]['1n1d'][len_size]['exec_time'] = []
                                    dict[file]['1n1d'][len_size]['torch_exec_time'] = []

                                start_pos = len("OneFlow AI-Writer time: ")
                                end_pos = line.find(" ", start_pos)
                                dict[file]['1n1d'][len_size]['exec_time'].append(float(line[start_pos:end_pos - 1]))


                with open(os.path.join(root_pytorch, file), "r", encoding="utf8") as f:     
                    count_i = 0                 
                    torch_exec_time = []
                    lines = copy.deepcopy(list(f.readlines()))
                    lines_copy = copy.deepcopy(lines)
                    for i, pytorch_line in enumerate(lines):
                        if "PyTorch AI-Writer time: " in pytorch_line: 
                            count_i += 1
                            if count_i <= 10:
                                continue

                            if not pytorch_line.startswith("PyTorch AI-Writer time: "):
                                pytorch_line = pytorch_line[pytorch_line.find("PyTorch AI-Writer time: "):]
                            
                            # if 1n1d or not
                            if "output_preLen=" in pytorch_line:
                                # catch length
                                len_start_pos = pytorch_line.find('=') + 1
                                len_size = pytorch_line[len_start_pos:-1]

                                start_pos = len("PyTorch AI-Writer time: ")
                                end_pos = pytorch_line.find(" ", start_pos)
                                dict[file]['1n1d'][len_size]['torch_exec_time'].append(float(pytorch_line[start_pos:end_pos - 1]))


                for k in dict[file].keys():
                    for key, value in dict[file][k].items():
                        
                        value['exec_time'].remove(min(value['exec_time']))
                        value['exec_time'].remove(max(value['exec_time']))

                        value['torch_exec_time'].remove(min(value['torch_exec_time']))
                        value['torch_exec_time'].remove(max(value['torch_exec_time']))

                        value['mean_time'] = str(np.around(np.mean(value['exec_time']), 3)) + "s"
                        value['mean_torch_time'] = str(np.around(np.mean(value['torch_exec_time']), 3)) + "s"
                        value['mean_relative_speed'] = np.around(float(value['mean_torch_time'][:-1]) / float(value['mean_time'][:-1]), 3)
                        
                        file_end_point = file.rfind('_')
                        nsys_root = 'https://oneflow-test.oss-cn-beijing.aliyuncs.com/EagerTest/.../oneflow/'
                        nsys_file = 'AI_Writer_eager_outlen%s_ws%s_' % (key, k[2]) + file[18:file_end_point] + '.qdrep'
                        value['nsys'] = nsys_root + nsys_file

        
        for file_name in dict.keys():
            for key, value in dict[file_name]['1n1d'].items():

                print(len(value['exec_time']), len(value['torch_exec_time']))
                result_line = "| AI-Writer_outLen%s_ws%s_%s |" % (key, '1', file_name[18:])

                result_line += (" %s / %s [nsys](%s)| " % (value['mean_time'], value['mean_relative_speed'], value['nsys']))
                # result_line += (" %s / %s | " % (value['mean_time'], value['mean_relative_speed']))
                result_line += (" %s | " % value['mean_torch_time'])

                process_res.write(result_line + "\n")