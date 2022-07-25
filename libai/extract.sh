COMMIT=${1} #set oneflow's commitID
rm -rf extract_log/libai_log
set -ex
LIBAI_LOG=extract_log/libai_log
MEGATRON_LOG=extract_log/megatron_log
mkdir -p $LIBAI_LOG
mkdir -p $MEGATRON_LOG

./ossutil64 cp -r oss://oneflow-test/OneFlowAutoTest/huoshanyingqin/baseline/megatron_base_supple/full/ ./$MEGATRON_LOG/
./ossutil64 cp -r oss://oneflow-test/OneFlowAutoTest/huoshanyingqin/$COMMIT/ ./$LIBAI_LOG/

python3 extract_bert.py --test-log $LIBAI_LOG --compare-log $MEGATRON_LOG --oneflow-commit $COMMIT

./ossutil64 cp -r -f $LIBAI_LOG/dlperf_result.md oss://oneflow-test/OneFlowAutoTest/huoshanyingqin/$COMMIT/
