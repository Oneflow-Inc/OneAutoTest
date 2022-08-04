# -*- coding: utf-8 -*-  

import sys
import os
import argparse

def print_rank_0(*args, **kwargs):
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module_name", type=str)
    parser.add_argument("--input_preLen", type=int, default=20)
    args = parser.parse_args()

    dict_input = {}
    dict_input["20"] = "我不知道自己想输入什么，但至少要到二十字"

    input = {
        "txt": dict_input[str(args.input_preLen)],
        "preLen": args.input_preLen # 预测的长度
    }

    sys.path.append('/home/xuyongning/writer/AI_Writer_Web')
    from AI_Writer_Web.infer import Writer as Writer_Oneflow
    writer_oneflow = Writer_Oneflow()
    output_oneflow = writer_oneflow.inference(input)
    oneflow_time = output_oneflow["time"]
    # print(output_oneflow["txt"])

    print_rank_0(
        f"OneFlow {args.module_name} time: {oneflow_time:.8f}s , input_preLen={args.input_preLen}"
    )

    sys.path.append('/home/xuyongning/writer/AI_Writer')
    from AI_Writer.infer import Writer as Writer_Pytorch
    writer_pytorch = Writer_Pytorch()
    output_pytorch = writer_pytorch.inference(input)
    pytorch_time = output_pytorch["time"]
    # print(output_pytorch["txt"])

    print_rank_0(
        f"PyTorch {args.module_name} time: {pytorch_time:.8f}s , input_preLen={args.input_preLen}"
    )

    relative_speed = pytorch_time / oneflow_time
    print_rank_0(
        f"Relative speed: {relative_speed:.8f} (= {pytorch_time:.8f}s / {oneflow_time:.8f}s)"
    )
