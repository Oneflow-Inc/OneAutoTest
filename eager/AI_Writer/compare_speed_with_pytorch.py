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
    parser.add_argument("--output_preLen", type=int, default=20)
    args = parser.parse_args()

    dict_input = {}
    dict_input["20"] = "我不知道自己想输入什么，但至少要到二十字"
    dict_input["512"] = "孩子们所盼望的，过年过节之外，大概要数迎神赛会的时候了。但我家的所在很偏僻，待到赛会的行列经过时，一定已在下午，仪仗之类，也减而又减，所剩的极其寥寥。往往伸着颈子等候多时，却只见十几个人抬着一个金脸或蓝脸红脸的神像匆匆地跑过去。于是，完了。我常存着这样的一个希望：这一次所见的赛会，比前一次繁盛些。可是结果总是一个“差不多”；也总是只留下一个纪念品，就是当神像还未抬过之前，化一文钱买下的，用一点烂泥，一点颜色纸，一枝竹签和两三枝鸡毛所做的，吹起来会发出一种刺耳的声音的哨子，叫作“吹都都”的，吡吡地吹它两三天。现在看看《陶庵梦忆》，觉得那时的赛会，真是豪奢极了，虽然明人的文章，怕难免有些夸大。因为祷雨而迎龙王，现在也还有的，但办法却已经很简单，不过是十多人盘旋着一条龙，以及村童们扮些海鬼。那时却还要扮故事，而且实在奇拔得可观。他记扮《水浒传》中人物云：“……于是分头四出，寻黑矮汉，寻梢长大汉，寻头陀，寻胖大和尚，寻茁壮妇人，寻姣长妇人，寻青面，寻歪头，寻赤须，寻美髯，寻黑大汉，寻赤脸长须。大索城中；无，则之郭，之村，之山僻，之邻府州县。用重价聘之，得三十六人，梁山泊好汉，个个呵活，臻臻至至，人马称娖而行。……”这"

    input = {
        "txt": dict_input["512"],
        "preLen": args.output_preLen # 预测的长度
    }

    sys.path.append('/home/xuyongning/writer/AI_Writer_Web')
    from AI_Writer_Web.infer import Writer as Writer_Oneflow
    writer_oneflow = Writer_Oneflow()
    output_oneflow = writer_oneflow.inference(input)
    oneflow_time = output_oneflow["time"]
    # print(output_oneflow["txt"])

    print_rank_0(
        f"OneFlow {args.module_name} time: {oneflow_time:.8f}s , output_preLen={args.output_preLen}"
    )

    sys.path.append('/home/xuyongning/writer/AI_Writer')
    from AI_Writer.infer import Writer as Writer_Pytorch
    writer_pytorch = Writer_Pytorch()
    output_pytorch = writer_pytorch.inference(input)
    pytorch_time = output_pytorch["time"]
    # print(output_pytorch["txt"])

    print_rank_0(
        f"PyTorch {args.module_name} time: {pytorch_time:.8f}s , output_preLen={args.output_preLen}"
    )

    relative_speed = pytorch_time / oneflow_time
    print_rank_0(
        f"Relative speed: {relative_speed:.8f} (= {pytorch_time:.8f}s / {oneflow_time:.8f}s)"
    )
