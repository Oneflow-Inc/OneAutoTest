# -*- coding: utf-8 -*-  

import sys
import os
import json
import argparse

def print_rank_0(*args, **kwargs):
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module_name", type=str)
    parser.add_argument("--output_preLen", type=int, default=20)
    args = parser.parse_args()
    
    with open("input_text.json", encoding="utf-8") as f:
        dict_input = json.load(f)
    
    input = {
        "txt": dict_input[str(args.output_preLen)],
        "preLen": args.output_preLen # 预测的长度
    }
    
    sys.path.append('/path/to/writer/AI_Writer_Web')
    from AI_Writer_Web.infer import Writer as Writer_Oneflow
    writer_oneflow = Writer_Oneflow()
    output_oneflow = writer_oneflow.inference(input)
    oneflow_time = output_oneflow["time"]
    #print(output_oneflow["txt"])

    print_rank_0(
        f"OneFlow {args.module_name} time: {oneflow_time:.8f}s , output_preLen={args.output_preLen}"
    )
    

    sys.path.append('/path/to/writer/AI_Writer')
    from AI_Writer.infer import Writer as Writer_Pytorch
    writer_pytorch = Writer_Pytorch()
    output_pytorch = writer_pytorch.inference(input)
    
    pytorch_time = output_pytorch["time"]
    #print(output_pytorch["txt"])

    print_rank_0(
        f"PyTorch {args.module_name} time: {pytorch_time:.8f}s , output_preLen={args.output_preLen}"
    )
    
    relative_speed = pytorch_time / oneflow_time
    print_rank_0(
        f"Relative speed: {relative_speed:.8f} (= {pytorch_time:.8f}s / {oneflow_time:.8f}s)"
    )
    