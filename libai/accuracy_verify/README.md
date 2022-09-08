## 使用方法

参考 https://github.com/Oneflow-Inc/libai/pull/316#issuecomment-1198826336
### 准备环境

    ```bash    
    git clone git@github.com:Oneflow-Inc/libai.git && 在libai/engine/trainer.py里添加显存输出代码
    cd libai
    cd tools && 拷贝 args_libai_bert_init.sh args_libai_bert_loss.sh args_libai_gpt2_init.sh args_libai_gpt2_loss.sh args_libai_t5_init.sh args_libai_t5_loss.sh
    cd configs && 拷贝 OneAutoTest/libai/bert_nl24_nah16_hs1024.py OneAutoTest/libai/gpt2_nl24_nah16_hs1024.py OneAutoTest/libai/t5_nl12_nah12_hs768.py && 如果是25-28机，需添一句 train.rdma_enabled=False
    cd .. && 拷贝 init.sh loss.sh draw_loss.py compose.py
    mkdir loss_txt && mkdir curve

    # 分别为 master对照组 和 测试分支 创建虚拟环境
    conda create -n acc python=3.8 && conda activate acc && 安装GradAcc测试分支对应的oneflow
    conda create -n master python=3.8 && conda activate master && 安装对照组master分支对应的onflow
    ```

### 生成模型初始化权重
- 显存设置为iter=1时输出，不要注释掉checkpointer
    ```bash
    bash init.sh
    ```
- 在test_logs_init路径下会保存初始化模型

### 跑loss对齐的一键测试
- 注意，如果只跑acc分支的测试，不跑master对照组，则需要将`args_libai_bert_loss.sh` L95的done删掉，并在L57加上
    ```python
    done

    TEST_COMMIT=${ACC_COMMIT}
    ```
    `args_libai_gpt2_loss.sh` 和 `args_libai_t5_loss.sh` 做相同修改
- 显存设置为iter=99时输出，注释掉checkpointer
- 把 `libai/data/build.py` 中 `persistent_workers=True if num_workers > 0 else False` 全部注释掉, 必要时把所有的shuffle都设置为False
- 在libai/engine/trainer.py下加保存loss的语句
    ```python
    total_losses_reduced = sum(metrics_dict.values())
    if dist.is_main_process():
        txt = open("loss_txt/your_loss.txt", "a")
        txt.write(str(total_losses_reduced.item())+"\n")
    ```
- 运行
    ```bash
    bash loss.sh "c4ce8fb" # 替换成测试commit
    ```
- 在loss_txt路径下保存有loss的数据，curve路径下有对齐的png图像，test_logs_loss路径下有训练日志
- 整理曲线图

    把曲线图按模型上传到oss上，再拷贝一份到本地，修改`compose.py`中的 `IMAGES_PATH` `IMAGE_SAVE_PATH` 和 `IMAGE_COLUMN`，并运行 `python3 compose.py`
- 整理显存及吞吐数据

    把test_logs_loss路径下的数据上传至oss，修改`extract_libai_libai.py`中的oss路径名，运行 `python extract_libai_libai.py --compare-log ./test_logs_loss/master/ --test-log ./test_logs_loss/c4ce8fb/`

## 测试结果

https://github.com/Oneflow-Inc/OneTeam/issues/1670

