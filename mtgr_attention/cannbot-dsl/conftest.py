"""原测试脚本pytest入口配置。"""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_addoption(parser):
    parser.addoption('--mtgr-npu', action='store_true')
    parser.addoption('--mtgr-output', default='')
    parser.addoption('--mtgr-mode', choices=['accuracy','compare','acceptance'], default='acceptance')
    parser.addoption('--mtgr-cases', default='')
    parser.addoption('--mtgr-reference-cache', action='append', default=[])
    parser.addoption('--mtgr-workers', type=int, default=32)
    parser.addoption('--mtgr-generalization', type=int, default=0)
    parser.addoption('--mtgr-acceptance-report', default='')

collect_ignore = ['run_test.py']
