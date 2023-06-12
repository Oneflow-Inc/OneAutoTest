export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_GDR_LEVEL=2
# 安装 TCCL 之后不需要 NCCL TOPO 文件
# export NCCL_TOPO_FILE=/data_turbo/home/workspace/nccl-tests/nccl_topo_a800_1.6t.xml
export NCCL_IB_QPS_PER_CONNECTION=4
export ONEFLOW_COMM_NET_IB_GID_INDEX=3
#export ONEFLOW_COMM_NET_IB_HCA=$NCCL_IB_HCA
export ONEFLOW_COMM_NET_IB_HCA=mlx5_bond_1:1
