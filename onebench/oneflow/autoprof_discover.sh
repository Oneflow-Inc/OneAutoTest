
# /bash/bin
set -x

RE_CLONE=${1:-true}

python3 -m pip uninstall -y oneflow 

python3 -m pip install flowvision

python3 -m pip install torch torchvision torchaudio
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117


if [ ! -d "./oneflow_autoprof_discover" ]; then
  mkdir oneflow_autoprof_discover
fi

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,noheader)"
GPU_NAME="${GPU_NAME// /_}"
ONEFLOW_COMMIT=$(python3 -c 'import oneflow; print(oneflow.__git_commit__)')
ONEFLOW_VERSION=$(python3 -c 'import oneflow; print(oneflow.__version__)')


if $RE_CLONE; then
  rm -rf ./oneflow_autoprof_discover/oneflow
fi

git clone --depth 1 https://github.com/Oneflow-Inc/oneflow.git ./oneflow_autoprof_discover/oneflow
cd ./oneflow_autoprof_discover/oneflow

TEST_ONEFLOW_COMMIT=$(git log --pretty=format:"%H" -n 1)
LOG_FILENAME=autoprof/${GPU_NAME}/$(date "+%Y%m%d")/${ONEFLOW_COMMIT}/

export ONEFLOW_PROFILE_CSV=oneflow_autoprof_discover_${GPU_NAME}_$(date "+%Y%m%d")_${ONEFLOW_COMMIT}

python3 -m oneflow.autoprof discover ./python/oneflow/test/modules/  2>&1 | tee ${ONEFLOW_PROFILE_CSV}.log

echo "oneflow-autoprof-discover(test_info)=$TEST_ONEFLOW_COMMIT" >> ${ONEFLOW_PROFILE_CSV}.log
echo "oneflow-version(test_info)=$ONEFLOW_VERSION" >> ${ONEFLOW_PROFILE_CSV}.log

~/ossutil64 cp -f -r ${ONEFLOW_PROFILE_CSV}.*  oss://oneflow-test/OneAutoTest/onebench/oneflow/${LOG_FILENAME}/




