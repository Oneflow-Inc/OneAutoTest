export ONEFLOW_COMM_NET_IB_GID_INDEX=$NCCL_IB_GID_INDEX
export ONEFLOW_COMM_NET_IB_HCA=mlx5_2:1

NNODES=$WORLD_SIZE
# GPUS_PER_NODE=8
NODE_RANK=$RANK

MASTER_ADDRS=${MASTER_ADDR}
MASTER_PORTS=${MASTER_PORT}
