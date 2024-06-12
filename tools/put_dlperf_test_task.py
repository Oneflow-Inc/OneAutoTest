import json
import os
import oss2
import time


def get_oss_auth(endpoint="oss-cn-beijing.aliyuncs.com", bucket="oneflow-test"):
    ki = os.getenv("OSS_ACCESS_KEY_ID")
    ks = os.getenv("OSS_ACCESS_KEY_SECRET")
    auth = oss2.Auth(ki, ks)
    return oss2.Bucket(auth, endpoint, bucket)


def get_task_info(commit, task_name):
    bucket_obj = get_oss_auth()
    local_task_path = "{}/debug_dlperf_task.json".format(os.environ["HOME"])
    bucket_obj.get_object_to_file(
        "OneBrain/Binary/debug_dlperf_task.json", local_task_path
    )
    with open(local_task_path) as f:
        task_info = json.load(f)
    print(task_info)
    if commit not in task_info.keys():
        task_info[commit] = {}
        task_info[commit][task_name] = {}
        task_info[commit][task_name]["status"] = 0  # 0表示未开始，1表示运行完成
        task_info[commit][task_name]["throughput"] = 0
        task_info[commit][task_name]["memory"] = 0
        task_info = json.dumps(task_info)
        with open(local_task_path, "w",) as f:
            f.writelines(task_info)
        bucket_obj.put_object_from_file(
            "OneBrain/Binary/debug_dlperf_task.json", local_task_path
        )
        print(1)

    print(0)

    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", type=str, required=True)
    parser.add_argument(
        "--task_name",
        type=str,
        required=False,
        default="ResNet50_lazy_DCgpu_FP16_b512_1n1g",
    )

    args = parser.parse_args()
    get_task_info(args.commit, args.task_name)
