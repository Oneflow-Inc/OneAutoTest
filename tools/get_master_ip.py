import json


def get_master_ip():
    str = ""
    with open("./hostfile.json", "r", encoding="utf-8") as f:
        temp = json.loads(f.read())
    for item in temp:
        if item["role"] == "master":
            str = item["ip"]
    print(str)


if __name__ == "__main__":
    get_master_ip()
