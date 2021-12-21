import json
import socket


def get_node_rank():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    i = 0
    with open("./hostfile.json", "r", encoding="utf-8") as f:
        temp = json.loads(f.read())
        for item in temp:
            if item["ip"] == ip:
                break
            else:
                i += 1
        print(i)


if __name__ == "__main__":
    get_node_rank()
