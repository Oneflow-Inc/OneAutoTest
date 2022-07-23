#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# python3 -m pip install requests pycryptodome pybase64 datetime
# python3 task/get_token.py --onebrain-server ${ONEBRAIN_SERVER} --grant-type ${ONEBRAIN_GRANT_TYPE} --client-id ${ONEBRAIN_CLIENT_ID} --client-secret ${ONEBRAIN_CLIENT_SECRET} --public-key ${ONEBRAIN_PUBLIC_KEY} --onebrain-username ${ONEBRAIN_USERNAME} --onebrain-password ${ONEBRAIN_PASSWORD} --onebrain-project-id ${ONEBRAIN_PROJECT_ID}
import time

from common import run_task,get_token,get_auto_test_list
from config import get_args

if __name__ == "__main__":
    args = get_args()
    headers = get_token(args)
    task_list = get_auto_test_list(
        args.onebrain_server, headers, args.onebrain_project_id
    )
    for task in task_list:
        if task["name"] == "ResNet50-dlperf-1n1g":
            run_task(args.onebrain_server, task["id"], headers=headers)
            time.sleep(600)

    exit(0)
