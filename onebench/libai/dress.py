import math
import matplotlib.pyplot as plt
import json
import os
import click


@click.command()
@click.option('--path', prompt='Your path', help='Path of your floater')

def print_loss(path):
    total_x = []
    master_loss = []
    naive_loss = []
    rank_loss = []
    name = ""
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0 and dirs[-1].startswith('LibAI'):
            name = dirs[-1]
        for file in files:
            if file.endswith('.json'):
                if "master" in root:
                    with open(os.path.join(root, file), 'r', encoding="utf-8", newline="\n") as f:
                        for line in f.readlines():
                            dev_data = json.loads(line)
                            total_x.append(dev_data["iteration"])
                            master_loss.append(dev_data["total_loss"])
                if "naive" in root:
                    with open(os.path.join(root, file), 'r', encoding="utf-8", newline="\n") as f:
                        for line in f.readlines():
                            dev_data = json.loads(line)
                            naive_loss.append(dev_data["total_loss"])
                if "rank" in root:
                    with open(os.path.join(root, file), 'r', encoding="utf-8", newline="\n") as f:
                        for line in f.readlines():
                            dev_data = json.loads(line)
                            rank_loss.append(dev_data["total_loss"])

    step_names = ["", "50-220", "100-220"]
    step = [0, 50, 100]
    for i, s in enumerate(step):
        plt.figure(figsize=(10, 8), dpi=120)
        plt.plot(total_x[s:], rank_loss[s:], label="rank_per_process")
        plt.plot(total_x[s:], naive_loss[s:], label="naive", linestyle="--")
        plt.plot(total_x[s:], master_loss[s:], label="master", linestyle=":")

        plt.xlabel("iteration")
        plt.ylabel("total_loss")
        plt.title(name[62:])
        plt.legend()  # 打上标签
        plt.savefig("./" + name + step_names[i] + ".png")


if __name__ == '__main__':
    print_loss()