# 动态Shape验证报告

验证日期：2026-11-25至26日。

Ascend950DT；NPU0；32 Cube / 64 AIV；CANN 9.3.0；torch 2.12.0+cpu；torch_npu 2.12.0.post2；cannbot-dsl 0.8.3。

独立Host动态编译缓存与固定metadata容量测试17/17通过；此项不等同于NPU功能验证。另有真实ACLNN功能514组及10000结构泛化全部通过，覆盖动态长度、batch、段数、GQA和核数等配置，详情见精度报告。仍按D=32/64/128三种编译变体复用动态规格。
