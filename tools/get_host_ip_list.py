import json


def get_host_ip_list():
    str = ""
    with open("./hostfile.json", "r", encoding="utf-8") as f:
        temp = json.loads(f.read())
        for item in temp:
            if str == "":
                str = item["ip"]
            else:
                str = str + "," + item["ip"]

    print(str)


if __name__ == "__main__":
    get_host_ip_list()
