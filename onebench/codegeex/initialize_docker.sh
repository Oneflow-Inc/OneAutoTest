#!/bin/bash

DOCKER_IMAGE="nvcr.io/nvidia/pytorch:21.11-py3"
DOCKER_NAME=$(openssl rand -hex 10)
PORT=$(shuf -i 8000-9999 -n 1)

docker pull $DOCKER_IMAGE
docker run -p $PORT:5002 --cpus 12 --gpus '"device=0"' -it -d --ipc=host --name=$DOCKER_NAME -v $(pwd):/workspace $DOCKER_IMAGE
docker cp /data/home/codegeex_13b.pt $DOCKER_NAME:/workspace/
docker cp /data/home/ouyangyu/codegeex/codegeex-fastertransformer/codegeex_13b_ft.pt $DOCKER_NAME:/workspace/
docker exec -it $DOCKER_NAME /bin/bash