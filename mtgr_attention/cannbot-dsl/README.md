# Cannbot DSL MTGR Ragged Segment Attention

仓库：[idiot-1217/mtgr_attention](https://github.com/idiot-1217/mtgr_attention)

布局与相邻tilelang-ascend一致：算子代码、golden、testcase和run_test位于本目录，results保存报告和结果。

详细说明见[交付报告](results/交付报告.md)、[精度报告](results/精度报告.md)、[性能报告](results/性能报告.md)。支持Ascend950DT、BF16、H=8、D=32/64/128、GQA=1/2/4/8；其余约束见算子接口校验和testcase。

## 环境与执行

安装匹配CANN的torch、torch_npu及cannbot-dsl，并由部署方提供已安装的ACLNN运行库和算子资源包。交付目录只提供运行测试入口。

```bash
cd /path/to/cannbot-dsl
source /path/to/cann/set_env.sh
export MTGR_ACLNN_LIBRARY=/path/to/installed/libmtgr_aclnn.so
export ASCEND_CUSTOM_OPP_PATH=/path/to/installed/opp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 只测一条配置：原始case464，三个种子，真实ACLNN调用。
python run_test.py --mode accuracy --case-index 464 --output-dir "$PWD/results/smoke"
# 全量468条功能 + 七条性能（需msprof/asys；性能仅占主kernel）。
python -m pytest test_attention.py -q -s --mtgr-npu --mtgr-mode acceptance --mtgr-output "$PWD/results/acceptance"
# 基础验收完成后才能运行泛化。
python run_test.py --mode accuracy --generalization 10000 --batch-size 128 --acceptance-report "$PWD/results/acceptance/validation_report.txt" --output-dir "$PWD/results/generalization"
```

本机验证根目录默认是脚本所在目录，可用MTGR_VALIDATION_ROOT指定结果允许落盘的父目录。设备固定NPU0；公共锁/tmp/cannbotdsl-validation-device0.lock应由管理员提供。每个功能设备子任务最多300秒，性能最多180秒；CPU计算在锁外完成。

原生运行库由部署环境提供，未打入Git。results中的历史绝对路径是实测溯源信息，执行脚本已改为基于自身目录解析，不引用原samples/test仓库布局。

接口、Host和Kernel已合并为mtgr_ragged_segment_attention.py；metadata和common独立保留。测试脚本通过aclnn_backend.py调用预安装ACLNN运行库，不回退到Python attention。
