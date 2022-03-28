#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# python3 -m pip install requests pycryptodome pybase64 datetime
# python3 task/get_token.py --onebrain-server ${ONEBRAIN_SERVER} --grant-type ${ONEBRAIN_GRANT_TYPE} --client-id ${ONEBRAIN_CLIENT_ID} --client-secret ${ONEBRAIN_CLIENT_SECRET} --public-key ${ONEBRAIN_PUBLIC_KEY} --onebrain-username ${ONEBRAIN_USERNAME} --onebrain-password ${ONEBRAIN_PASSWORD} --onebrain-project-id ${ONEBRAIN_PROJECT_ID}
import requests, base64, time

from datetime import datetime
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
from config import get_args


def encrypt(txt, pulick_key):
    pulick_key = base64.standard_b64decode(bytes(pulick_key, encoding="utf8")).decode(
        "utf-8"
    )
    pub_key = RSA.importKey(pulick_key)
    cipher = PKCS1_cipher.new(pub_key)
    ciphertext = cipher.encrypt(txt.encode("utf-8"))

    return base64.standard_b64encode(ciphertext).decode("utf-8")


def get_token(args):

    payload = {
        "grant_type": args.grant_type,
        "client_id": args.client_id,
        "username": args.onebrain_username,
        "client_secret": encrypt(args.client_secret, args.public_key),
        "password": encrypt(args.onebrain_password, args.public_key),
    }

    ob_result = requests.post(
        args.onebrain_server + "/api/serv-auth/oauth/token", params=payload
    )
    try:
        print(ob_result.json())
        access_token = ob_result.json()["data"]["access_token"]
        return {"Authorization": "Bearer %s" % access_token}
    except:
        print("Get token failed")
        print(ob_result)
        exit(1)


def get_auto_test_list(host_url, headers, project_id):
    try:
        ob_result = requests.get(
            host_url
            + "/api/serv-admin/api/1/task/pageList?projectId={}&current=1&size=100".format(
                project_id
            ),
            headers=headers,
        )

        return ob_result.json()["result"]["result"]
    except:
        print("Get task list failed")
        print(ob_result)
        exit(1)


def run_task(host_url, task_id, headers):
    requests.put(
        host_url + "/api/serv-admin/api/1/task/start/{}".format(task_id),
        headers=headers,
    )


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
