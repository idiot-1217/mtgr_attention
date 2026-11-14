"""ACLNN 测试适配器：创建真实 aclTensor，调用两段式 C ABI，不回退 Python attention。

按原Python参数顺序直传C ABI；metadata=None由原生executor生成，本模块不补齐或重排参数。
"""

import ctypes as ct
from functools import lru_cache
import inspect
import os

import torch

from mtgr_ragged_segment_attention import mtgr_ragged_segment_attention as python_interface


@lru_cache(maxsize=1)
def libraries():
    """加载用户指定的真实 ACLNN 库并声明 C 函数签名。

    Returns:
        tuple: ACL tensor API、workspace 查询函数及执行函数。
    """
    library = ct.CDLL(os.environ['MTGR_ACLNN_LIBRARY'], mode=ct.RTLD_GLOBAL)
    tensor_api = ct.CDLL('libnnopbase.so', mode=ct.RTLD_GLOBAL)
    tensor_api.aclCreateTensor.restype = ct.c_void_p
    tensor_api.aclCreateTensor.argtypes = [ct.POINTER(ct.c_int64), ct.c_uint64, ct.c_int,
        ct.POINTER(ct.c_int64), ct.c_int64, ct.c_int, ct.POINTER(ct.c_int64), ct.c_uint64, ct.c_void_p]
    tensor_api.aclDestroyTensor.argtypes = [ct.c_void_p]
    prepare = library.aclnnMtgrRaggedSegmentAttentionGetWorkspaceSize
    prepare.restype = ct.c_int
    signature = inspect.signature(python_interface)
    prepare.argtypes = [ct.c_void_p if 'Tensor' in str(parameter.annotation) else
        ct.c_double if parameter.annotation is float else ct.c_int64
        for parameter in signature.parameters.values()] + [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_void_p)]
    execute = library.aclnnMtgrRaggedSegmentAttention
    execute.restype = ct.c_int
    execute.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
    return tensor_api, prepare, execute


def precompile(config, data, core_num=32):
    """在设备锁外检查预安装的 ACLNN 运行库。

    Args:
        config: 当前用例配置。
        data: CPU 输入数据。
        core_num: Cube 核数，支持 1～32。

    Returns:
        None: 检查运行库路径，不执行设备计算。
    """
    assert 1 <= core_num <= 32
    from pathlib import Path
    assert Path(os.environ['MTGR_ACLNN_LIBRARY']).is_file()


def mtgr_ragged_segment_attention(*args, **kwargs):
    """通过真实 ACLNN 两段式 API 同步执行一次测试调用。

    Args:
        args: 与 Python attention 接口一致的位置参数。
        kwargs: 与 Python attention 接口一致的关键字参数。

    Returns:
        torch.Tensor: 原地更新的调用方输出。

    Raises:
        RuntimeError: aclTensor 创建、workspace 查询或 ACLNN 执行失败。
    """
    bound = inspect.signature(python_interface).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    query = values['query']
    tensor_api, prepare, execute = libraries()
    descriptors = []
    workspace = None
    try:
        arguments = []
        for name, parameter in inspect.signature(python_interface).parameters.items():
            value = values[name]
            if 'Tensor' not in str(parameter.annotation):
                arguments.append(value)
                continue
            if value is None:
                arguments.append(None)
                continue
            dimensions = (ct.c_int64 * value.ndim)(*value.shape)
            strides = (ct.c_int64 * value.ndim)(*value.stride())
            descriptor = tensor_api.aclCreateTensor(dimensions, value.ndim,
                27 if value.dtype == torch.bfloat16 else 3, strides, 0, 2,
                dimensions, value.ndim, value.data_ptr())
            if not descriptor:
                raise RuntimeError('Failed to create an ACLNN tensor descriptor.')
            descriptors.append(descriptor)
            arguments.append(descriptor)
        workspace_size, executor = ct.c_uint64(), ct.c_void_p()
        status = prepare(*arguments, ct.byref(workspace_size), ct.byref(executor))
        if status:
            raise RuntimeError(f'ACLNN workspace query failed with status {status}.')
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=query.device)
        stream = torch.npu.current_stream(query.device.index)
        status = execute(workspace.data_ptr(), workspace_size.value, executor, stream.npu_stream)
        if status:
            raise RuntimeError(f'ACLNN execution failed with status {status}.')
        stream.synchronize()
        print(f'ACLNN_C_ABI success workspace_bytes={workspace_size.value}', flush=True)
    finally:
        for descriptor in descriptors:
            tensor_api.aclDestroyTensor(descriptor)
    return values['output_snd']
