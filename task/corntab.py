#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# python3 -m pip install requests pycryptodome pybase64 datetime
# python3 task/get_token.py --onebrain-server ${ONEBRAIN_SERVER} --grant-type ${ONEBRAIN_GRANT_TYPE} --client-id ${ONEBRAIN_CLIENT_ID} --client-secret ${ONEBRAIN_CLIENT_SECRET} --public-key ${ONEBRAIN_PUBLIC_KEY} --onebrain-username ${ONEBRAIN_USERNAME} --onebrain-password ${ONEBRAIN_PASSWORD} --onebrain-project-id ${ONEBRAIN_PROJECT_ID}
import time
from datetime import datetime
from common import run_task,get_token,get_auto_test_list
from config import get_args


if __name__ == "__main__":
    args = get_args()
    headers = get_token(args)
    task_list = get_auto_test_list(
        args.onebrain_server, headers, args.onebrain_project_id
    )
    train_1n8g_task_id = ""
    train_2n4g_task_id = ""
    for task in reversed(task_list):
        if (
            "train" in task["name"]
            and task["gpuNum"] == 8
            and len(task["taskSlaveList"]) == 0
        ):
            print("train task", task["name"])
            train_1n8g_task_id = task["id"]
        elif "train" in task["name"] and "2n4g" in task["name"]:
            print("train task", task["name"])
            train_2n4g_task_id = task["id"]

        elif task["gpuNum"] in [1, 8] and len(task["taskSlaveList"]) == 0:
            print("1n1g", task["name"])
            time.sleep(50)
            run_task(args.onebrain_server, task["id"], headers=headers)

    day_of_week = datetime.now().weekday()
    if day_of_week == 5 and train_1n8g_task_id != "":
        time.sleep(3600)
        run_task(args.onebrain_server, train_1n8g_task_id, headers=headers)
    if day_of_week == 6 and train_2n4g_task_id != "":
        time.sleep(3600)
        run_task(args.onebrain_server, train_2n4g_task_id, headers=headers)

    exit(0)
