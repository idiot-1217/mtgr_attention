# 平铺交付目录冒烟验证

## 交付目录路径检查及推送前冒烟（2026-11-14）

已从本地`D:\cpp_code\cann-dsl\tilelang_mtgr_attention\mtgr_attention\cannbot-dsl`制作完整快照，在NPU开发机`/mnt/map/l00946857/.cache/mtgr-delivery-layout-20261114/cannbot-dsl`镜像运行。所有18个算子/测试源文件逐一按LF归一化SHA256核对一致，Windows本机无NPU执行环境，因此设备验证使用该同内容镜像。库沿用昨日通过验收的12变体libmtgr_aclnn.so，kernel源码未修改；本轮验证目录重排及脚本路径适配，不将其计作重新构建原生库的验证。

命令：`python run_test.py --mode accuracy --case-index 464 --workers 3 --output-dir "$PWD/results/layout_smoke"`。case464三个随机种子0/1/2均通过原L0双golden检查，无INF/NAN、逐位确定性及零行检查均通过，退出码0；本次参考全部现算。详细日志、实际环境、逐种子结果和源码签名见`layout_smoke/`。这是一条新增冒烟用例，不替代2026-11-13的468条/514组全量与10000条泛化结果，也未重新采集七条性能。

路径修改仅涉及run_test.py的模块搜索与输出根目录、validation_support.py的源码/签名根目录。均基于`__file__`定位；结果默认写入交付目录内，可由MTGR_VALIDATION_ROOT显式配置。原golden、case配置、核心算子实现保持不变。本次使用部署环境中的预安装运行库。
