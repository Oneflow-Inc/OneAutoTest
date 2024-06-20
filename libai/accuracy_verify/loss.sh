
MASTER_COMMIT=${1:-"ecafd61b09349a1c6c45333ea6eff96009cf66c0"}
ACC_COMMIT=${2:-"3d5e919cb700d84f52d4cf2730083931f17a91bb"}
BRANCH=${3:-"dev_cc_acc_mem_v5"}

for TEST_COMMIT in ${MASTER_COMMIT} ${ACC_COMMIT}
do
if [ $TEST_COMMIT != $MASTER_COMMIT ];then
    COMMIT=${ACC_COMMIT:0:7}
    python3 -m pip uninstall oneflow -y
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/canary/refs/heads/${BRANCH}/cu112/index.html
else
    COMMIT="master"
    python3 -m pip uninstall oneflow -y
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112/${TEST_COMMIT}
fi
pip install -e .

bash examples/acc_loss.sh ${COMMIT}

done
