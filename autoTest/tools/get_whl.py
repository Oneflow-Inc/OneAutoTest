import os
import oss2
import json


def get_whl_from_oss_by_commit(args, path):
    auth = oss2.Auth("LTAIE7QOeCBmk6xx", "1AJGkjJK3QixNS1MoXedwVOQvRWiAJ")
    bucket_obj = oss2.Bucket(auth, args.endpoint, args.bucket)

    tmp_url = ""
    files = bucket_obj.list_objects(path)
    for f in files.object_list:
        if args.py_version in f.key:
            tmp_url = bucket_obj.sign_url("GET", f.key, 3600, slash_safe=True)
            break

    return tmp_url


def put_task_to_oss(
    local_task_path, endpoint="oss-cn-beijing.aliyuncs.com", bucket="oneflow-test"
):
    auth = oss2.Auth("LTAIE7QOeCBmk6xx", "1AJGkjJK3QixNS1MoXedwVOQvRWiAJ")
    bucket_obj = oss2.Bucket(auth, endpoint, bucket)
    bucket_obj.put_object_from_file(
        "OneBrain/Binary/debug_dlperf_task.json", local_task_path
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        required=False,
        default="oss-cn-beijing.aliyuncs.com",
    )
    parser.add_argument("--bucket", type=str, required=False, default="oneflow-staging")
    parser.add_argument("--task_path", type=str, required=True)
    parser.add_argument("--py_version", type=str, required=False, default="cp37")
    parser.add_argument("--cuda_version", type=str, required=False, default="cu112")
    parser.add_argument("--branch_name", type=str, required=False, default="master")

    args = parser.parse_args()
    with open(args.task_path) as f:
        task_info = json.load(f)
    for commit in task_info:
        if task_info[commit]["ResNet50_lazy_DCgpu_FP16_b512_1n1g"]["status"] == 0:
            path = "canary/commit/{}/{}/oneflow-".format(commit, args.cuda_version)
            tmp_url = get_whl_from_oss_by_commit(args, path)
            if tmp_url == "":
                path = "branch/{}/{}/{}/oneflow-".format(
                    args.branch_name, args.cuda_version, commit
                )
                tmp_url = get_whl_from_oss_by_commit(args, path)
            task_info[commit]["ResNet50_lazy_DCgpu_FP16_b512_1n1g"]["status"] = 1

    print(tmp_url)
    task_info = json.dumps(task_info)
    with open(args.task_path, "w",) as f:
        f.writelines(task_info)
    put_task_to_oss(args.task_path)


