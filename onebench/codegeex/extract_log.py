import os
import re
import numpy as np
import argparse

def process_logs(log_files_prefix, num_runs, is_faster_transformer=False):
    memory_usage = []
    process_code_time = []

    for i in range(1, num_runs + 1):
        with open(f"{log_files_prefix}_{i}.log", "r") as f:
            content = f.read()

            mem = re.search(r"\d+/\d+/\d+ \d+:\d+:\d+\.\d+, NVIDIA A100-PCIE-40GB, \d+\.\d+\.\d+, \d+ %, \d+ %, \d+ MiB, \d+ MiB, (\d+) MiB", content)
            if is_faster_transformer:
                time = re.search(r"process_code time used (\d+\.\d+)", content)
            else:
                time = re.search(r"Total generation time: (\d+\.\d+)", content)

            if mem and time:
                memory_usage.append(int(mem.group(1)))
                process_code_time.append(float(time.group(1)))

    return np.mean(memory_usage), np.mean(process_code_time)

def main(logs_path, framework_list):
    lengths = [128, 256, 512, 1024, 2048]
    num_runs = 10
    framework_list = ["oneflow", "pytorch", "faster_transformer"]

    results = {}

    for length in lengths:
        results[length] = {}

        for framework in framework_list:
            log_files_prefix = os.path.join(logs_path_dict[framework], f"{length}_{framework}_run")
            avg_memory, avg_time = process_logs(log_files_prefix, num_runs, is_faster_transformer=(framework == "faster_transformer"))
            results[length][framework] = (avg_memory, avg_time)

    markdown_table = "| L | OneFlow[Mem(MiB)/Time(s)] | PyTorch[Mem(MiB)/Time(s)] | FasterTransformer[Mem(MiB)/Time(s)] |\n| --- | --- | --- | --- |\n"

    for length, framework_results in results.items():
        row = f"| {length} | {framework_results['oneflow'][0]:.2f}/{framework_results['oneflow'][1]:.3f} | {framework_results['pytorch'][0]:.2f}/{framework_results['pytorch'][1]:.3f} | {framework_results['faster_transformer'][0]:.2f}/{framework_results['faster_transformer'][1]:.3f} |\n"
        markdown_table += row

    with open("results_table.md", "w") as f:
        f.write(markdown_table)

    print(markdown_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oneflow_logs_path", type=str, required=True, help="Path to the OneFlow log files")
    parser.add_argument("--pytorch_logs_path", type=str, required=True, help="Path to the PyTorch log files")
    parser.add_argument("--faster_transformer_logs_path", type=str, required=True, help="Path to the FasterTransformer log files")
    args = parser.parse_args()

    logs_path_dict = {
        "oneflow": args.oneflow_logs_path,
        "pytorch": args.pytorch_logs_path,
        "faster_transformer": args.faster_transformer_logs_path,
    }

    main(logs_path_dict)