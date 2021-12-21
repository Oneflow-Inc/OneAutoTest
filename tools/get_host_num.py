import json


def get_host_num():
    str = ""
    with open("./hostfile.json", "r", encoding="utf-8") as f:
        temp = json.loads(f.read())
        print(len(temp))


if __name__ == "__main__":
    get_host_num()
