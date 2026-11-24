"""动态编译缓存与固定metadata容量的Host侧回归检查。"""
from pathlib import Path
import sys
import pytest


@pytest.fixture
def dynamic_host():
    """加载可用的DSL环境；未安装设备依赖时跳过。"""
    pytest.importorskip('torch')
    pytest.importorskip('cannbotdsl')
    source = Path(__file__).resolve().parent
    if not (source / 'mtgr_ragged_segment_attention.py').is_file():
        source = Path(__file__).resolve().parents[3] / 'samples/attention/mtgr_ragged_segment_attention'
    sys.path.insert(0, str(source))
    import mtgr_ragged_segment_attention as host
    return host


def test_ragged_group_and_core_share_compile_cache(dynamic_host, monkeypatch):
    """不同ragged形状、GQA及核数必须复用同一个编译结果。"""
    calls = []
    executable = object()
    def compile_once(*arguments):
        calls.append(arguments)
        return executable
    monkeypatch.setattr(dynamic_host.dsl, 'compile', compile_once)
    dynamic_host._compile_mtgr_dynamic_executable.cache_clear()
    try:
        for total, batch, segments, pages, columns in ((308, 1, 2, 1, 3), (32104, 8, 1027, 17, 62)):
            for group in (1, 2, 4, 8):
                for cores in (8, 16, 28, 32):
                    result = dynamic_host.build_mtgr_ragged_segment_attention_executable(
                        total, 128, group, batch, segments, pages, columns, cores=cores
                    )
                    assert result is executable
        assert len(calls) == 1
    finally:
        dynamic_host._compile_mtgr_dynamic_executable.cache_clear()


@pytest.mark.parametrize('cores', [8, 16, 28, 32])
@pytest.mark.parametrize('kv_heads', [1, 2, 4, 8])
def test_metadata_capacity_is_independent_of_active_cores(dynamic_host, cores, kv_heads):
    """固定容量计划应适配全部核数，旧的按核数分配长度应被拒绝。"""
    import torch
    query = torch.empty((1, 8, 128), dtype=torch.bfloat16)
    metadata = torch.empty((16 + kv_heads * 36 * 16,), dtype=torch.int32)
    tiling = dynamic_host.MtgrRaggedSegmentAttentionTiling()
    tiling.validate_metadata(metadata, query, 1, kv_heads, cores)
    old_metadata = torch.empty((16 + kv_heads * cores * 16,), dtype=torch.int32)
    with pytest.raises(ValueError):
        tiling.validate_metadata(old_metadata, query, 1, kv_heads, cores)
