# Cannbot DSL MTGR Ragged Segment Attention

仓库：[idiot-1217/mtgr_attention](https://github.com/idiot-1217/mtgr_attention)

布局与相邻tilelang-ascend一致：算子代码、golden、testcase和run_test位于本目录，results保存报告和结果。

详细说明见[交付报告](results/交付报告.md)、[精度报告](results/精度报告.md)、[性能报告](results/性能报告.md)。支持Ascend950DT、BF16、H=8、D=32/64/128、GQA=1/2/4/8；其余约束见算子接口校验和testcase。

## 环境与执行

安装匹配CANN的torch、torch_npu及cannbot-dsl，并由部署方提供已安装的ACLNN运行库和算子资源包。交付目录只提供运行测试入口。

```bash
# 从交付目录执行；事先加载CANN环境并设置运行库和算子资源环境变量。
: "${MTGR_ACLNN_LIBRARY:?请配置ACLNN运行库环境变量}"
: "${ASCEND_CUSTOM_OPP_PATH:?请配置算子资源环境变量}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 只测一条配置：原始case464，三个种子，真实ACLNN调用。
python run_test.py --mode accuracy --case-index 464 --output-dir "results/smoke"
# 全量468条功能 + 七条性能（需msprof/asys；性能仅占主kernel）。
python -m pytest test_attention.py -q -s --mtgr-npu --mtgr-mode acceptance --mtgr-output "results/acceptance"
# 基础验收完成后才能运行泛化。
python run_test.py --mode accuracy --generalization 10000 --batch-size 128 --acceptance-report "results/acceptance/validation_report.txt" --output-dir "results/generalization"
```

本机验证根目录默认是脚本所在目录，可用MTGR_VALIDATION_ROOT指定结果允许落盘的父目录。设备固定NPU0；共享设备锁应由管理员提供。每个功能设备子任务最多300秒，性能最多180秒；CPU计算在锁外完成。

原生运行库由部署环境提供，未打入Git。results中的历史绝对路径是实测溯源信息，执行脚本已改为基于自身目录解析，不引用原samples/test仓库布局。

接口、Host和Kernel已合并为mtgr_ragged_segment_attention.py；metadata和common独立保留。测试脚本通过aclnn_backend.py调用预安装ACLNN运行库，不回退到Python attention。

交付只保存报告和汇总CSV。命令中的validation_report.txt由测试执行时生成，用于后续泛化的前置验收检查，不是随仓库保存的文件。

## 动态Shape支持（2026-11-24）

支持五个ragged动态维、动态KV头数以及运行时核数。相同D和分块复用同一编译结果，不再因请求shape、GQA或核数变化重编；原生运行包保留D=32/64/128三种特化，每种特化支持GQA=1/2/4/8。

metadata按36核槽容量分配，实际支持核数仍为1～32。长度为16 + batch × kv_heads × 36 × 16个INT32元素；计划内容按实际核数连续排布，尾部空间不参与计算。显式metadata必须使用此容量规格，旧长度计划需要重新生成。

运行环境升级为cannbot-dsl 0.8.3、CANN 9.3.0。随附runtime/mtgr_runtime_ascend950.tar.gz包含预编译运行库、metadata库和算子资源。解压后，将MTGR_ACLNN_LIBRARY指向其中lib目录的运行库，将ASCEND_CUSTOM_OPP_PATH指向opp目录，再运行上述测试命令。运行测试不需要重新编译原生库。

新增Host回归可执行`python -m pytest test_dynamic_shape.py -q`。完整新增能力和性能结果见[动态Shape验证报告](results/动态Shape验证报告.md)。
