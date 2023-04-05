import math
import matplotlib.pyplot as plt
import json
import os
import click

@click.command()
@click.option('--path', prompt='Your path', help='Path of your floater')
def print_loss(path):
    total_x = []
    total_y = []
    name = ""
    for root, dirs, files in os.walk(path):
        if len(dirs) > 0 and dirs[-1].startswith('LibAI'):
            name = dirs[-1]
        for file in files:
            
            if file.endswith('.json'):

                try:
                    with open(os.path.join(root, file), 'r', encoding="utf-8", newline="\n") as f:
                        com_x = []
                        dev_y = []
                        for line in f.readlines():
                            dev_data = json.loads(line)
                            
                            dev_y.append(dev_data['total_loss'])
                            com_x.append(dev_data["iteration"])

                except Exception as e:
                    print(f"Error opening file {os.path.join(root, file)}: {e}")
        
                total_y.append(dev_y)
                total_x.append(com_x)

    new_total_y = []
    new_total_x = []
    for i in range(len(total_y)):
        if len(total_y[i]) >= 20:
            new_total_y.append(total_y[i])
            new_total_x.append(total_x[i])

    plt.figure(figsize=(10, 8), dpi=120)
    for i in range(0, len(total_y), 3):
        plt.plot(new_total_x[i],new_total_y[i],label="rank_per_process")
        plt.plot(new_total_x[i + 1],new_total_y[i + 1],label="master",linestyle = "--")
        plt.plot(new_total_x[i + 2],new_total_y[i + 2],label="naive",linestyle = ":")
        plt.xlabel("iteration")
        plt.ylabel("total_loss")
        plt.title(name[62:])
        plt.legend()   #打上标签
        plt.savefig("./" + name + ".png")

    plt.figure(figsize=(10, 8), dpi=120)
    for i in range(0, len(total_y), 3):
        plt.plot(new_total_x[i][50:],new_total_y[i][50:],label="rank_per_process")
        plt.plot(new_total_x[i + 1][50:],new_total_y[i + 1][50:],label="master",linestyle = "--")
        plt.plot(new_total_x[i + 2][50:],new_total_y[i + 2][50:],label="naive",linestyle = ":")
        plt.xlabel("iteration")
        plt.ylabel("total_loss")
        plt.title(name[62:])
        plt.legend()   #打上标签
        plt.savefig("./" + name + "_50-220.png")

    plt.figure(figsize=(10, 8), dpi=120)
    for i in range(0, len(total_y), 3):
        plt.plot(new_total_x[i][100:],new_total_y[i][100:],label="rank_per_process")
        plt.plot(new_total_x[i + 1][100:],new_total_y[i + 1][100:],label="master",linestyle = "--")
        plt.plot(new_total_x[i + 2][100:],new_total_y[i + 2][100:],label="naive",linestyle = ":")
        plt.xlabel("iteration")
        plt.ylabel("total_loss")
        plt.title(name[62:])
        plt.legend()   #打上标签
        plt.savefig("./" + name + "._100-220.png")

if __name__ == '__main__':
    print_loss()
