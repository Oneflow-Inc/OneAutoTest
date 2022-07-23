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
    bucket_obj.get_object_to_file("Binary/debug_dlperf_task.json", local_task_path)
    with open(local_task_path) as f:
        task_info = json.load(f)
    if commit not in task_info.keys():
        print(0)
    else:
        while task_info[commit][task_name]["status"] != 1:
            time.sleep(600)
            bucket_obj.get_object_to_file(
                "Binary/debug_dlperf_task.json", local_task_path
            )
            with open(local_task_path) as f:
                task_info = json.load(f)

    print(task_info[commit][task_name]["throughput"])

    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)

    args = parser.parse_args()
    get_task_info(args.commit, args.task_name)
