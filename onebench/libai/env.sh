ENABLE_NCCL_LOGICAL_FUSION=${1:-false}
ONEFLOW_GRAPH_NCCL_LOGICAL_FUSION_BUCKET_SIZE=${2:-0}

export ENABLE_NCCL_LOGICAL_FUSION=$ENABLE_NCCL_LOGICAL_FUSION
export ONEFLOW_GRAPH_NCCL_LOGICAL_FUSION_BUCKET_SIZE=$ONEFLOW_GRAPH_NCCL_LOGICAL_FUSION_BUCKET_SIZE

echo ENABLE_NCCL_LOGICAL_FUSION=$ENABLE_NCCL_LOGICAL_FUSION
echo ONEFLOW_GRAPH_NCCL_LOGICAL_FUSION_BUCKET_SIZE=$ONEFLOW_GRAPH_NCCL_LOGICAL_FUSION_BUCKET_SIZE
