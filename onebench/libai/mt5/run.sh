set -ex

TEST_BRANCH=${1:-"main"}

#python3 -m pip uninstall -y oneflow 

#python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112/ee077040fbd240ca3b35492a0c224fc17bffc271/index.html

if [ ! -d "./libai" ]; then
  git clone --depth 1 https://github.com/Oneflow-Inc/libai.git
fi


if [ $TEST_BRANCH != 'main' ]; then
  git remote set-branches origin $TEST_BRANCH
  git fetch --depth 1 origin $TEST_BRANCH
  git checkout $TEST_BRANCH
fi

if [ ! -d "./libai/projects/T5/data/training_data" ]; then
  cd ./libai/projects/T5
  mkdir -p ./data/
  wget -nc http://oneflow-static.oss-cn-beijing.aliyuncs.com/libai/mt5_init/wudao_180g_test_bert_tokenized_512_demo.zip
  unzip -n -d ./data/ wudao_180g_test_bert_tokenized_512_demo.zip
  mv ./data/wudao_180g_test_bert_tokenized_512_train ./data/training_data
  cd -
fi

wget -nc https://raw.githubusercontent.com/Oneflow-Inc/OneAutoTest/main/onebench/libai/mt5/args_t5_mt5.sh -P ./libai/tools/

cd libai

python3 -m pip install -e . --user

# NNODES GPUS_PER_NODE 
bash tools/args_t5_mt5.sh 1 1 0 127.0.0.1 1 1 true true 4 16


