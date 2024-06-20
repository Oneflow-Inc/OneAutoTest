
MASTER_COMMIT=${1:-"ecafd61b09349a1c6c45333ea6eff96009cf66c0"}

python3 -m pip uninstall oneflow -y
python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112/${MASTER_COMMIT}
pip install -e .

bash examples/acc_init.sh
